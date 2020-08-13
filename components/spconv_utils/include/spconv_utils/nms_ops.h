// Created by Zhiliang Zhou on 2020
// this is a simplified version of spconv
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

#ifndef SPCONV_UTILS_NMS_OPS_H
#define SPCONV_UTILS_NMS_OPS_H
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace spconv {
namespace py = pybind11;

template <typename DType>
std::vector<int> non_max_suppression_cpu(py::array_t<DType> boxes,
                                         py::array_t<int> order,
                                         DType thresh,
                                         DType eps = 0);

template <typename DType>
std::vector<int> rotate_non_max_suppression_cpu(py::array_t<DType> box_corners,
                                                py::array_t<int> order,
                                                py::array_t<DType> standup_iou,
                                                DType thresh);

template <typename DType>
int non_max_suppression(py::array_t<DType> boxes,
                        py::array_t<int> keep_out,
                        DType nms_overlap_thresh,
                        int device_id);

}  // namespace spconv

#endif //SPCONV_UTILS_NMS_OPS_H
