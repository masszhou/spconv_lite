// Created by Zhiliang Zhou on 2020
// this is a simplified version of spconv
// changes:
// 1. use TORCH_LIBRARY macro instead of torch::RegisterOperators(). new feature in pytorch 1.6
//
// original spconv was implemented by Yan Yan, https://github.com/traveller59/spconv
// -------------------------------------------------------------------
// Copyright 2019 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "spconv_cpp/spconv_ops.h"
#include "spconv_cpp/pool_ops.h"
#include <torch/custom_class.h>

// >>usage:
// torch.ops.load_library("build/libwarp_perspective.so")
// help(torch.ops.spconv_cpp.get_indice_pairs())
// >>explanation:
// access torch.ops.namespace.function in Python
// here spconv_cpp is namespace, get_indice_pairs is function
TORCH_LIBRARY(spconv_cpp, m) {
    m.def("get_indice_pairs", &spconv::getIndicePairs);
    m.def("indice_conv", &spconv::indiceConv);
    m.def("indice_conv_backward", &spconv::indiceConvBackward);
    m.def("indice_maxpool", &spconv::indiceMaxPool);
    m.def("indice_maxpool_backward", &spconv::indiceMaxPoolBackward);
}