//
// Created by zzhou on 07.08.20.
//

# include "spconv_utils/point2voxel.h"

namespace spconv{

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
        int max_voxels)
{
    // e.g. auto r = x.mutable_unchecked<3>(); // Will throw if ndim != 3 or flags.writeable is false
    auto points_rw = points.template mutable_unchecked<2>();
    // here template is a keyword for mutable_unchecked NOT for .
    // without template keyword points.mutable_unchecked<2>() can be interpreted as
    // ((points.mutable_unchecked) < 2) > () for compiler
    // more reading https://stackoverflow.com/questions/8463368/template-dot-template-construction-usage
    auto voxels_rw = voxels.template mutable_unchecked<3>();
    auto voxel_point_mask_rw = voxel_point_mask.template mutable_unchecked<2>();
    auto coors_rw = coors.mutable_unchecked<2>();
    auto num_points_per_voxel_rw = num_points_per_voxel.mutable_unchecked<1>();
    auto coor_to_voxelidx_rw = coor_to_voxelidx.mutable_unchecked<NDim>();

    // number of input points
    auto N = points_rw.shape(0);
    // number of features
    auto num_features = points_rw.shape(1);
    // auto ndim = points_rw.shape(1) - 1;
    constexpr int ndim_minus_1 = NDim - 1;
    int voxel_num = 0;
    bool failed = false;
    int coor[NDim];
    int c;
    int grid_size[NDim];

    for (int i = 0; i < NDim; ++i) {
        grid_size[i] = round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i]);
    }
    int voxelidx, num;
    for (int i = 0; i < N; ++i) {
        failed = false;
        for (int j = 0; j < NDim; ++j) {
            c = floor((points_rw(i, j) - coors_range[j]) / voxel_size[j]);
            if ((c < 0 || c >= grid_size[j])) {
                failed = true;
                break;
            }
            coor[ndim_minus_1 - j] = c;
        }
        if (failed)
            continue;
        voxelidx = coor_to_voxelidx_rw(coor[0], coor[1], coor[2]);
        if (voxelidx == -1) {
            voxelidx = voxel_num;
            if (voxel_num >= max_voxels)
                continue;
            voxel_num += 1;
            coor_to_voxelidx_rw(coor[0], coor[1], coor[2]) = voxelidx;
            for (int k = 0; k < NDim; ++k) {
                coors_rw(voxelidx, k) = coor[k];
            }
        }
        num = num_points_per_voxel_rw(voxelidx);
        if (num < max_points) {
            voxel_point_mask_rw(voxelidx, num) = DType(1);
            for (int k = 0; k < num_features; ++k) {
                voxels_rw(voxelidx, num, k) = points_rw(i, k);
            }
            num_points_per_voxel_rw(voxelidx) += 1;
        }
    }
    for (int i = 0; i < voxel_num; ++i) {
        coor_to_voxelidx_rw(coors_rw(i, 0), coors_rw(i, 1), coors_rw(i, 2)) = -1;
    }
    return voxel_num;
}

// Explicitly instantiate templates
template int points_to_voxel_3d_np<double, 3>(
            py::array_t<double> points,
            py::array_t<double> voxels,
            py::array_t<double> voxel_point_mask,
            py::array_t<int> coors,
            py::array_t<int> num_points_per_voxel,
            py::array_t<int> coor_to_voxelidx,
            std::vector<double> voxel_size,
            std::vector<double> coors_range,
            int max_points,
            int max_voxels);

template int points_to_voxel_3d_np<float, 3>(
        py::array_t<float> points,
        py::array_t<float> voxels,
        py::array_t<float> voxel_point_mask,
        py::array_t<int> coors,
        py::array_t<int> num_points_per_voxel,
        py::array_t<int> coor_to_voxelidx,
        std::vector<float> voxel_size,
        std::vector<float> coors_range,
        int max_points,
        int max_voxels);

}  // namespace spconv