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

#ifndef SPCONV_UTILS_BOX_IOU_H
#define SPCONV_LITE_BOX_IOU_H
#include <pybind11/pybind11.h>
// must include pybind11/eigen.h if using eigen matrix as arguments.
#include <pybind11/numpy.h>

namespace spconv {
namespace py = pybind11;

template <typename DType, typename ShapeContainer>
inline py::array_t<DType> constant(ShapeContainer shape, DType value) {
    // create ROWMAJOR array.
    py::array_t<DType> array(shape);
    std::fill(array.mutable_data(), array.mutable_data() + array.size(), value);
    return array;
}


template <typename DType>
inline py::array_t<DType> zeros(std::vector<long int> shape) {
    return constant<DType, std::vector<long int>>(shape, 0);
}

template <typename DType>
py::array_t<DType>
rbbox_iou(py::array_t<DType> box_corners,
          py::array_t<DType> qbox_corners,
          py::array_t<DType> standup_iou,
          DType standup_thresh);

template <typename DType>
py::array_t<DType>
rbbox_intersection(py::array_t<DType> box_corners,
                   py::array_t<DType> qbox_corners,
                   py::array_t<DType> standup_iou,
                   DType standup_thresh);

}

#endif //SPCONV_LITE_BOX_IOU_H
