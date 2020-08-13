#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "spconv_utils/nms_ops.h"
#include "spconv_utils/box_iou.h"
#include "spconv_utils/point2voxel.h"

namespace py = pybind11;
using namespace pybind11::literals;

// package name "spconv_utils_cpp" should be the same with library name in CMake
PYBIND11_MODULE(spconv_utils_cpp, m) {
    m.doc() = "pybind11 utils functions for spconv";
    m.def("points_to_voxel_3d_np", &spconv::points_to_voxel_3d_np<float, 3>, "matrix tensor_square", 
        py::arg("points") = 1,
        py::arg("voxels") = 2,
        py::arg("voxel_point_mask") = 3, 
        py::arg("coors") = 4, 
        py::arg("num_points_per_voxel") = 5,
        py::arg("coor_to_voxelidx") = 6, 
        py::arg("voxel_size") = 7, 
        py::arg("coors_range") = 8,
        py::arg("max_points") = 9, 
        py::arg("max_voxels") = 10);
      // if using namespace pybind11::literals; can be shorten as
    m.def("points_to_voxel_3d_np", &spconv::points_to_voxel_3d_np<double, 3>, "matrix tensor_square", 
        "points"_a = 1, 
        "voxels"_a = 2,
        "voxel_point_mask"_a = 3, 
        "coors"_a = 4, 
        "num_points_per_voxel"_a = 5,
        "coor_to_voxelidx"_a = 6, 
        "voxel_size"_a = 7, 
        "coors_range"_a = 8,
        "max_points"_a = 9, 
        "max_voxels"_a = 10);
    m.def("non_max_suppression", &spconv::non_max_suppression<double>, 
        py::return_value_policy::reference_internal, 
        "bbox iou",
        py::arg("boxes") = 1,
        py::arg("keep_out") = 2, 
        py::arg("nms_overlap_thresh") = 3, 
        py::arg("device_id") = 4);
    m.def("non_max_suppression", &spconv::non_max_suppression<float>,
        py::return_value_policy::reference_internal, 
        "bbox iou", 
        py::arg("boxes") = 1,
        py::arg("keep_out") = 2, 
        py::arg("nms_overlap_thresh") = 3, 
        py::arg("device_id") = 4);
    m.def("non_max_suppression_cpu", &spconv::non_max_suppression_cpu<double>,
          py::return_value_policy::reference_internal,
          "bbox iou",
          "boxes"_a = 1,
          "order"_a = 2,
          "nms_overlap_thresh"_a = 3,
          "eps"_a = 4);
    m.def("non_max_suppression_cpu", &spconv::non_max_suppression_cpu<float>,
          py::return_value_policy::reference_internal,
          "bbox iou",
          "boxes"_a = 1,
          "order"_a = 2,
          "nms_overlap_thresh"_a = 3,
          "eps"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &spconv::rotate_non_max_suppression_cpu<float>,
          py::return_value_policy::reference_internal,
          "bbox iou",
          "box_corners"_a = 1,
          "order"_a = 2,
          "standup_iou"_a = 3,
          "thresh"_a = 4);
    m.def("rotate_non_max_suppression_cpu", &spconv::rotate_non_max_suppression_cpu<double>,
          py::return_value_policy::reference_internal,
          "bbox iou",
          "box_corners"_a = 1,
          "order"_a = 2,
          "standup_iou"_a = 3,
          "thresh"_a = 4);
    m.def("rbbox_iou", &spconv::rbbox_iou<double>,
          py::return_value_policy::reference_internal,
          "rbbox iou",
          "box_corners"_a = 1,
          "qbox_corners"_a = 2,
          "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
    m.def("rbbox_iou", &spconv::rbbox_iou<float>,
          py::return_value_policy::reference_internal,
          "rbbox iou",
          "box_corners"_a = 1,
          "qbox_corners"_a = 2,
          "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
    m.def("rbbox_intersection", &spconv::rbbox_intersection<double>,
          py::return_value_policy::reference_internal,
          "rbbox iou",
          "box_corners"_a = 1,
          "qbox_corners"_a = 2,
          "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
    m.def("rbbox_intersection", &spconv::rbbox_intersection<float>,
          py::return_value_policy::reference_internal,
          "rbbox iou",
          "box_corners"_a = 1,
          "qbox_corners"_a = 2,
          "standup_iou"_a = 3,
          "standup_thresh"_a = 4);
}