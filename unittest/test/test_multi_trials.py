# Created by Zhiliang Zhou on 2020
# a simplified version of spconv
# changes
# 1. testSpConv3d with param ['cuda:0', [19, 18, 17], 1, 64, 32, k=2, s=2, p=0, d=1]: gradient w.r.t. input test failed in multi trials
#
# original implemented by Yan Yan, https://github.com/traveller59/spconv
# ----------------------------------------------------------------
# Copyright 2019 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import torch
import unittest
from tqdm import tqdm

from .utils import TestCase, generate_sparse_data, params_grid, gather_nd, scatter_nd
from .cases import SparseConv3dTestTorch
from .cases import Conv3dTestTorch
from .cases import SparseDeConv3dTestTorch
from .cases import DeConv3dTestTorch
from .cases import SparseMaxPoolTestTorch
from .cases import MaxPool3dTestTorch


class TestSpConvMultiTrials(TestCase):
    def testSpConv3d(self):
        """
        param ['cuda:0', [19, 18, 17], 1, 64, 32, k=2, s=2, p=0, d=1]: gradient w.r.t. input test failed in multi trials
        """
        print("\n-> test spconv_lite.SparseConv3d()")
        print("compare gradients and outputs with nn.Conv3d() with specified parameter set in multi trials")
        np.random.seed(484)
        devices = ["cuda:0"]
        shapes = [[19, 18, 17]]
        batchsizes = [1, 2]
        in_channels = [64]
        out_channels = [32, 48, 64]
        ksizes = [3]  # k=2, s=2, d=1 gradient w.r.t. input test failed in multi trials
        strides = [1, 2, 3]
        paddings = [0, 1, 2]
        dilations = [1, 2, 3]

        param_comb = params_grid(devices, shapes, batchsizes, in_channels, out_channels,
                                 ksizes, strides, paddings, dilations)

        # discard unsupported parameters
        param_comb = [[dev, shape, bs, IC, OC, k, s, p, d]
                      for dev, shape, bs, IC, OC, k, s, p, d in param_comb
                      if not all([s > 1, d > 1])]  # don't support this.

        for i in range(100):
            pbar = tqdm(total=len(param_comb))
            for each in param_comb:
                pbar.set_description(f"param {each}")
                dev, shape, bs, IC, OC, k, s, p, d = each
                device = torch.device(dev)
                num_points = [1000] * bs

                sparse_dict = generate_sparse_data(shape, num_points, IC)

                features = np.ascontiguousarray(sparse_dict["features"]).astype(np.float32)  # (1000, 64)
                indices = np.ascontiguousarray(sparse_dict["indices"][:, [3, 0, 1, 2]]).astype(np.int32)  # (1000, 4)
                features_dense = sparse_dict["features_dense"].astype(np.float32)  # (bn=1, ch=64, 19, 18, 17)

                filters = np.random.uniform(0, 1, size=[k, k, k, IC, OC]).astype(np.float32)
                indices_t = torch.from_numpy(indices).int().to(device)
                features_t = torch.from_numpy(features).to(device)  # (1000, 64)
                features_t.requires_grad = True
                features_dense_t = torch.from_numpy(features_dense).to(device)
                features_dense_t.requires_grad = True

                net = SparseConv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
                net_ref = Conv3dTestTorch(1, 3, shape, IC, OC, k, s, p, d).to(device)
                filters_t = torch.from_numpy(filters).to(device)
                net_ref.net[0].weight.data[:] = filters_t.permute(4, 3, 0, 1, 2).contiguous()
                net.net[0].weight.data[:] = filters_t

                out_ref = net_ref(features_dense_t)
                out = net(features_t, indices_t, bs).dense()

                # pseudo weights for non scalar loss
                dout = np.random.uniform(-0.2, 0.2, out_ref.shape).astype(features.dtype)
                dout_t = torch.from_numpy(dout).to(device)

                out.backward(dout_t)
                out_ref.backward(dout_t)

                # fetch gradient
                # reference value, gradient of input dense feature
                din_ref_dense = features_dense_t.grad.detach().permute(0, 2, 3, 4, 1).contiguous()  # [1,19,18,17, 64]
                # reference value, take the non-zero elements corresponding gradient of input dense feature
                din_ref_sparse = gather_nd(din_ref_dense, indices_t.long())  # [1000, 64]
                din_ref_sparse_np = din_ref_sparse.cpu().numpy()

                # test value, gradient of sparse tensor
                din_test_sparse = features_t.grad.detach()
                din_test_sparse_np = din_test_sparse.cpu().numpy()

                # compare gradient w.r.t. input
                self.assertAllClose(din_test_sparse_np, din_ref_sparse_np, atol=1e-4)

                # compare intermediate gradient
                for layer, layer_ref in zip(net.net, net_ref.net):
                    dw = layer.weight.grad.detach().cpu().numpy()
                    dw_ref = layer_ref.weight.grad.detach().cpu().numpy()
                    dw = dw.transpose(4, 3, 0, 1, 2)
                    self.assertAllClose(dw, dw_ref, atol=1e-4)

                # compare output
                out_np = out.detach().cpu().numpy()
                out_ref_np = out_ref.detach().cpu().numpy()
                self.assertAllClose(out_np, out_ref_np, atol=1e-4)
                pbar.update()
            pbar.close()


if __name__ == "__main__":
    unittest.main()