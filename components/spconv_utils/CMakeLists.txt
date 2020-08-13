cmake_minimum_required(VERSION 3.14)
project(spconv_utils LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------
# static lib: spconv_nms
# cuda code
add_library(spconv_nms STATIC
        src/nms.cu)
target_link_libraries(spconv_nms PUBLIC 
        ${CUDA_CUDART}
        )
target_include_directories(spconv_nms PRIVATE 
        include
        )
set_target_properties(spconv_nms PROPERTIES
        CXX_STANDARD 14
        CUDA_STANDARD 14
        POSITION_INDEPENDENT_CODE ON
        SOVERSION 1
        )

# ---------------------------------------------------------------------
# package content: spconv_utils_cpp
# expose to package
add_library(spconv_utils_cpp MODULE
        src/nms_ops.cpp
        src/box_iou.cpp
        src/point2voxel.cpp
        src/all.cpp
        )        
target_link_libraries(spconv_utils_cpp PRIVATE
        spconv_nms
        pybind11::module
        )
target_include_directories(spconv_utils_cpp PRIVATE
        include
        )
set_target_properties(spconv_utils_cpp PROPERTIES
        CXX_STANDARD 14
        CUDA_STANDARD 14
        POSITION_INDEPENDENT_CODE ON
        # for pybind11
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )