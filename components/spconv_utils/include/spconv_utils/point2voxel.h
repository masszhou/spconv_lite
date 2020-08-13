// Created by Zhiliang Zhou on 2020
// 
// Note:
// This file contains inline functions due to pybind11 mechanism
// either explicitly instantiate template
// or use big inline functions header
// 
// ------------------------------------------------------------------------
// modified from https://github.com/traveller59/spconv
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
#ifndef SPCONV_UTILS_POINT2VOXEL
#define SPCONV_UTILS_POINT2VOXEL

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace spconv {

namespace py = pybind11;

/**
 * @brief convert 3d points(N, >=3) to voxels.
 * @details a CPU function, convert 3d points(N, >=3) to voxels. This version calculate
 * everything in one loop. now it takes only 0.8ms(~6k voxels) with c++ and 3.2ghz cpu.
 *
 * input
 * @tparam DType, data type in numpy
 * @tparam NDim
 * @param points, [N, ndim] float tensor. points[:, :3] contain xyz points and
 *                points[:, 3:] contain other information such as reflectivity.
 * @param coor_to_voxelidx, int array. used as a dense map.
 * @param voxel_size, [3] list/tuple or array, float. xyz, indicate voxel size
 * @param coors_range, [6] list/tuple or array, float. indicate voxel range. format: xyzxyz, minmax
 * @param max_points, int e.g. =35, indicate maximum points contained in a voxel.
 * @param max_voxels, int e.g. =20000, indicate maximum voxels this function create.
 * Output
 * @param voxels, shape=(max_voxels, max_points, points.shape[-1])
 * @param coors, shape=(max_voxels, 3)
 * @param num_points_per_voxel, shape=(max_voxels, )
 * @param voxel_point_mask, shape=(max_voxels, max_points)
 * @return
 */
template<typename DType, int NDim>
int points_to_voxel_3d_np(
        py::array_t<DType> points,
        py::array_t<DType> voxels,
        py::array_t<DType> voxel_point_mask,
        py::array_t<int> coors,
        py::array_t<int> num_points_per_voxel,
        py::array_t<int> coor_to_voxelidx,
        std::vector<DType> voxel_size,
        std::vector<DType> coors_range,
        int max_points,
        int max_voxels);

} // namespace spconv

#endif //SPCONV_UTILS_POINT2VOXEL