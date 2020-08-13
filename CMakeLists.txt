cmake_minimum_required(VERSION 3.14)
project(spconv_lite LANGUAGES CXX CUDA VERSION 1.1)

# ---------------------------------------------------------------------
# general setup

add_compile_definitions(PYTORCH_VERSION=${PYTORCH_VERSION})
set(CMAKE_CXX_EXTENSIONS OFF) # avoid gnu++11 be added to CXX flags

# ---------------------------------------------------------------------
# libtorch setup
# get path from pytorch packaget, equivalent to 
# -DCMAKE_PREFIX_PATH="/home/zzhou/.virtualenvs/python3.8_spconv_lite/lib/python3.8/site-packages/torch"
find_package(Torch REQUIRED)
# fix nvcc fatal   : Unknown option '-Wall'
set_property(TARGET torch_cuda PROPERTY INTERFACE_COMPILE_OPTIONS "")
set_property(TARGET torch_cpu PROPERTY INTERFACE_COMPILE_OPTIONS "")

message(STATUS "Torch library status:")
message(STATUS "    CXX_FlAGS: ${TORCH_CXX_FLAGS}")
message(STATUS "    version: ${TORCH_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}")


# ---------------------------------------------------------------------
# CUDA setup
set(CUDA_LIB_PATH_HINTS "${CUDA_TOOLKIT_ROOT_DIR}/lib64")
find_library(CUDA_CUDART NAMES cudart HINTS ${CUDA_LIB_PATH_HINTS})
find_library(CUDA_CUBLAS NAMES cublas HINTS ${CUDA_LIB_PATH_HINTS})
# from torch package
torch_cuda_get_nvcc_gencode_flag(NVCC_FLAGS_EXTRA)
string (REPLACE ";" " " NVCC_FLAGS_EXTRA_STR "${NVCC_FLAGS_EXTRA}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${NVCC_FLAGS_EXTRA_STR}")

message(STATUS "CUDA library status:")
message(STATUS "    Found CUDA ${CUDA_VERSION_STRING} at ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "    CMAKE_CUDA_COMPILER: ${CMAKE_CUDA_COMPILER}")
message(STATUS "    CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "    CUDA_CUDART: ${CUDA_CUDART}")
message(STATUS "    CUDA_CUBLAS: ${CUDA_CUBLAS}")


# ---------------------------------------------------------------------
# pybind11 support
add_subdirectory(extern/pybind11)

# ---------------------------------------------------------------------
# tensorview GPU support, implementation idea is similar to torch::PackedTensorAccessor32
# by Yan Yan
add_compile_definitions(TV_CUDA)
add_library(tensorview INTERFACE)
target_include_directories(tensorview INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/tensorview/include>
        )

# ---------------------------------------------------------------------
# cuhash support
add_subdirectory(components/cuhash)

# ---------------------------------------------------------------------
# A C++ implementation of a fast hash map and hash set using robin hood hashing
# https://github.com/Tessil/robin-map
add_library(tsl INTERFACE)
target_include_directories(tsl INTERFACE
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/components/tsl/include>
        )

# ---------------------------------------------------------------------
# spconv_utils
add_subdirectory(components/spconv_utils)

# ---------------------------------------------------------------------
# spconv_cpp functions
add_subdirectory(components/spconv_cpp)