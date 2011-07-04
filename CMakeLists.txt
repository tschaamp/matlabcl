cmake_minimum_required(VERSION 2.8)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
find_package(OpenCL)
include_directories(${OPENCL_INCLUDE_DIRS})

find_package(Matlab)
include_directories(${MATLAB_INCLUDE_DIR})
include(MatlabMexMacro)

include_directories(src)

add_mex_file(plugin src/matlabcl.cc)
target_link_libraries(plugin ${OPENCL_LIBRARIES})
