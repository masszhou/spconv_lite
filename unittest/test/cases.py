# Created by Zhiliang Zhou on 2020
# a simplified version of spconv
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
from torch import nn
import spconv_lite as spconv


class Conv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding=padding,
                      dilation=dilation,
                      bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.Conv3d(out_channels,
                          out_channels,
                          kernel_size,
                          stride,
                          padding=padding,
                          dilation=dilation,
                          bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()
        layers = [
            spconv.SparseConv3d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding=padding,
                                dilation=dilation,
                                bias=False,
                                use_hash=False,
                                algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConv3d(out_channels,
                                    out_channels,
                                    kernel_size,
                                    stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False,
                                    use_hash=False,
                                    algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size, self.grid)
        return self.net(x)  # .dense()


class SubMConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size, stride, padding, dilation, algo=spconv.ConvAlgo.Native):
        super().__init__()
        layers = [
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False,
                              algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=False,
                                  algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # .cpu()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size, self.grid)
        return self.net(x)  # .dense()


class SparseDeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            spconv.SparseConvTranspose3d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding=padding,
                                         dilation=dilation,
                                         bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConvTranspose3d(out_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=False))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class DeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.ConvTranspose3d(out_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SparseDeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            spconv.SparseConvTranspose3d(in_channels,
                                         out_channels,
                                         kernel_size,
                                         stride,
                                         padding=padding,
                                         dilation=dilation,
                                         bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SparseConvTranspose3d(out_channels,
                                             out_channels,
                                             kernel_size,
                                             stride,
                                             padding=padding,
                                             dilation=dilation,
                                             bias=False))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class DeConv3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super().__init__()
        layers = [
            nn.ConvTranspose3d(in_channels,
                               out_channels,
                               kernel_size,
                               stride,
                               padding=padding,
                               dilation=dilation,
                               bias=False)
        ]
        for i in range(1, num_layers):
            layers.append(
                nn.ConvTranspose3d(out_channels,
                                   out_channels,
                                   kernel_size,
                                   stride,
                                   padding=padding,
                                   dilation=dilation,
                                   bias=False))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()


class SubMConv3dTestTorch(nn.Module):
    def __init__(self,
                 num_layers,
                 ndim,
                 shape,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 algo=spconv.ConvAlgo.Native):
        super().__init__()
        layers = [
            spconv.SubMConv3d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False,
                              algo=algo)
        ]
        for i in range(1, num_layers):
            layers.append(
                spconv.SubMConv3d(out_channels,
                                  out_channels,
                                  kernel_size,
                                  stride,
                                  padding=padding,
                                  dilation=dilation,
                                  bias=False,
                                  algo=algo))
        self.net = spconv.SparseSequential(*layers, )
        # self.grid = torch.full([3, *shape], -1, dtype=torch.int32).cuda()
        self.grid = None
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()  # .cpu()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size, self.grid)
        return self.net(x)  # .dense()


class SparseMaxPoolTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation):
        super().__init__()
        layers = [
            spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation)
        ]
        for i in range(1, num_layers):
            layers.append(spconv.SparseMaxPool3d(kernel_size, stride, padding, dilation))
        self.net = spconv.SparseSequential(*layers, )
        self.shape = shape

    def forward(self, features, coors, batch_size):
        coors = coors.int()
        x = spconv.SparseConvTensor(features, coors, self.shape, batch_size)
        return self.net(x)  # .dense()


class MaxPool3dTestTorch(nn.Module):
    def __init__(self, num_layers, ndim, shape, kernel_size, stride, padding,
                 dilation):
        super().__init__()
        layers = [nn.MaxPool3d(kernel_size, stride, padding, dilation)]
        for i in range(1, num_layers):
            layers.append(nn.MaxPool3d(kernel_size, stride, padding, dilation))
        self.net = nn.Sequential(*layers, )
        self.shape = shape

    def forward(self, x):
        return self.net(x)  # .dense()
