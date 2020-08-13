cmake_minimum_required(VERSION 3.14)
project(cuhash LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------
# lib: cuhash
add_library(cuhash STATIC)
target_sources(cuhash PRIVATE 
    src/hash_functions.cu 
    src/hash_table.cpp 
    src/hash_table.cu 
    src/hash_functions.cpp
)
target_link_libraries(cuhash PRIVATE tensorview)
target_include_directories(cuhash PUBLIC
    include)
target_include_directories(cuhash PRIVATE
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set_target_properties(cuhash PROPERTIES
    CXX_STANDARD 14
    CUDA_STANDARD 14
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)



# when make install
# INSTALL(TARGETS cuhash DESTINATION lib)
