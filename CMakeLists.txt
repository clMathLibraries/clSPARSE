project(clSPARSE)
cmake_minimum_required(VERSION 2.8)

# directory with external Find*.cmake files
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCL REQUIRED)
include_directories(SYSTEM ${OPENCL_INCLUDE_DIRS}) #make it visible in whole pro

message(STATUS "OpenCL: ${OPENCL_INCLUDE_DIRS}")

add_subdirectory(library)

#build tests for clSPARSE
set(BUILD_TESTS 1 CACHE TYPE BOOL)
if(BUILD_TESTS)
    add_subdirectory(tests)
endif(BUILD_TESTS)

set(BUILD_BENCHMARKS 1 CACHE TYPE BOOL)

if(BUILD_BENCHMARKS)
   add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)

