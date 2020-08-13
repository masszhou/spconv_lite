cmake_minimum_required(VERSION 3.14)
project(spconv_cpp LANGUAGES CXX CUDA)

find_package(OpenMP)

# ---------------------------------------------------------------------
# static lib: spconv_ops
# cuda/cpp code
add_library(spconv_ops STATIC
    src/indice.cpp
    src/indice.cu
    src/reordering.cpp
    src/reordering.cu
    src/spconv_ops.cpp
    )
# because tensorview is header only library, spconv_ops depends on tensorview, so it need nvcc to compile...
set_source_files_properties(src/spconv_ops.cpp PROPERTIES LANGUAGE CUDA)
target_link_libraries(spconv_ops
    PUBLIC
        ${CUDA_CUDART}
        ${CUDA_CUBLAS}
        ${TORCH_LIBRARIES}
        OpenMP::OpenMP_CXX  # used for CPU op
    PRIVATE
        cuhash
        tensorview
        tsl
    )
target_include_directories(spconv_ops PUBLIC
    include
    )
set_target_properties(spconv_ops PROPERTIES
    CXX_STANDARD 14
    CUDA_STANDARD 14
    POSITION_INDEPENDENT_CODE ON
    SOVERSION 1
    )

## ---------------------------------------------------------------------
## static lib: pool_ops
## cuda/cpp code
add_library(pool_ops STATIC
        src/maxpool.cpp
        src/maxpool.cu
        src/pool_ops.cpp
        )
target_link_libraries(pool_ops
        PUBLIC
            ${CUDA_CUDART}
            ${CUDA_CUBLAS}
            ${TORCH_LIBRARIES}
        PRIVATE
            tensorview
        )
target_include_directories(pool_ops PUBLIC
        include
        )
set_target_properties(pool_ops PROPERTIES
        CXX_STANDARD 14
        CUDA_STANDARD 14
        POSITION_INDEPENDENT_CODE ON
        SOVERSION 1
        )

# ---------------------------------------------------------------------
# package content: spconv_cpp
# expose to package
add_library(spconv_cpp MODULE
        src/all.cpp
        )
target_link_libraries(spconv_cpp PRIVATE
        spconv_ops
        pool_ops
        )
target_include_directories(spconv_cpp PRIVATE
        include
        )
set_target_properties(spconv_cpp PROPERTIES
        CXX_STANDARD 14
        CUDA_STANDARD 14
        POSITION_INDEPENDENT_CODE ON
        # for pybind11
#        PREFIX "${PYTHON_MODULE_PREFIX}"
#        SUFFIX "${PYTHON_MODULE_EXTENSION}"
        )