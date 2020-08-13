# Created by Zhiliang Zhou on 2020
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
from pathlib import Path
import torch
from enum import Enum

_LIB_FILE_NAME = "libspconv_cpp.so"
_LIB_PATH = str(Path(__file__).parent / _LIB_FILE_NAME)
torch.ops.load_library(_LIB_PATH)


class ConvAlgo(Enum):
    Native = 0  # small memory cost, faster when number of points is large.
    Batch = 1  # high memory cost, faster when number of points is small (< 50000)
    BatchGemmGather = 2  # high memory cost, faster when number of points medium


def get_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        size = (input_size[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) // stride[i] + 1
        if kernel_size[i] == -1:
            output_size.append(1)
        else:
            output_size.append(size)
    return output_size


def get_deconv_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    ndim = len(input_size)
    output_size = []
    for i in range(ndim):
        if kernel_size[i] == -1:
            raise ValueError("deconv don't support kernel_size < 0")
        size = (input_size[i] - 1) * stride[i] - 2 * padding[i] + kernel_size[i] + output_padding[i]
        output_size.append(size)
    return output_size


def get_indice_pairs(indices,
                     batch_size,
                     spatial_shape,
                     ksize=3,
                     stride=1,
                     padding=0,
                     dilation=1,
                     out_padding=0,
                     subm=False,
                     transpose=False,
                     grid=None,
                     use_hash=False):
    """
    function from spconv_ops.cpp, registered to torch.ops

    :param indices:
    :param batch_size:
    :param spatial_shape:
    :param ksize:
    :param stride:
    :param padding:
    :param dilation:
    :param out_padding:
    :param subm:
    :param transpose:
    :param grid:
    :param use_hash:
    :return:
    """
    ndim = indices.shape[1] - 1
    if not isinstance(ksize, (list, tuple)):
        ksize = [ksize] * ndim
    if not isinstance(stride, (list, tuple)):
        stride = [stride] * ndim
    if not isinstance(padding, (list, tuple)):
        padding = [padding] * ndim
    if not isinstance(dilation, (list, tuple)):
        dilation = [dilation] * ndim
    if not isinstance(out_padding, (list, tuple)):
        out_padding = [out_padding] * ndim

    for d, s in zip(dilation, stride):
        assert any([s == 1, d == 1]), "don't support this."

    if not subm:
        if transpose:
            out_shape = get_deconv_output_size(spatial_shape, ksize, stride,
                                               padding, dilation, out_padding)
        else:
            out_shape = get_conv_output_size(spatial_shape, ksize, stride,
                                             padding, dilation)
    else:
        out_shape = spatial_shape
    if grid is None:
        res = torch.ops.spconv_cpp.get_indice_pairs(indices,         # torch::Tensor indices
                                                    batch_size,      # int64_t batchSize
                                                    out_shape,       # std::vector<int64_t> outSpatialShape
                                                    spatial_shape,   # std::vector<int64_t> spatialShape
                                                    ksize,           # std::vector<int64_t> kernelSize
                                                    stride,          # std::vector<int64_t> stride
                                                    padding,         # std::vector<int64_t> padding
                                                    dilation,        # std::vector<int64_t> dilation
                                                    out_padding,     # std::vector<int64_t> outPadding
                                                    int(subm),       # int64_t _subM
                                                    int(transpose),  # int64_t _transpose
                                                    int(use_hash))   # int64_t _useHash
        return res
    else:
        raise NotImplementedError


def indice_conv(features,
                filters,
                indice_pairs,
                indice_pair_num,
                num_activate_out,
                inverse=False,
                subm=False,
                algo=ConvAlgo.Native.value):
    return torch.ops.spconv_cpp.indice_conv(features,          # torch::Tensor features
                                            filters,           # torch::Tensor filters
                                            indice_pairs,      # torch::Tensor indicePairs
                                            indice_pair_num,   # torch::Tensor indiceNum
                                            num_activate_out,  # int64_t numActOut
                                            int(inverse),      # int64_t _inverse
                                            int(subm),         # int64_t _subM
                                            algo)              # int64_t algo


def indice_conv_backward(features,
                         filters,
                         out_bp,
                         indice_pairs,
                         indice_pair_num,
                         inverse=False,
                         subm=False,
                         algo=ConvAlgo.Native.value):
    return torch.ops.spconv_cpp.indice_conv_backward(features,         # torch::Tensor features
                                                     filters,          # torch::Tensor filters
                                                     out_bp,           # torch::Tensor outGrad
                                                     indice_pairs,     # torch::Tensor indicePairs
                                                     indice_pair_num,  # torch::Tensor indiceNum
                                                     int(inverse),     # int64_t _inverse
                                                     int(subm),        # int64_t _subM
                                                     algo)             # int64_t algo


def indice_maxpool(features, indice_pairs, indice_pair_num, num_activate_out):
    return torch.ops.spconv_cpp.indice_maxpool(features,          # torch::Tensor features
                                               indice_pairs,      # torch::Tensor indicePairs
                                               indice_pair_num,   # torch::Tensor indiceNum
                                               num_activate_out)  # int64_t numAct


def indice_maxpool_backward(features, out_features, out_bp, indice_pairs, indice_pair_num):
    return torch.ops.spconv_cpp.indice_maxpool_backward(features,         # torch::Tensor features
                                                        out_features,     # torch::Tensor outFeatures
                                                        out_bp,           # torch::Tensor outGrad
                                                        indice_pairs,     # torch::Tensor indicePairs
                                                        indice_pair_num)  # torch::Tensor indiceNum


def nms(boxes, scores, pre_max_size, post_max_size, thresh, eps):
    res = torch.ops.spconv_cpp.nms(boxes,          # torch::Tensor boxes
                                   scores,         # torch::Tensor scores
                                   pre_max_size,   # int64_t preMaxSize
                                   post_max_size,  # int64_t postMaxSize
                                   thresh,         # double thresh
                                   eps)            # double eps
    return res