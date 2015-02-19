project(tests)
cmake_minimum_required(VERSION 2.8)

find_package(GTest REQUIRED)

find_package(Boost COMPONENTS
    program_options
)

set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} -std=c++11)
message(STATUS "GTest libraries:" ${GTEST_LIBRARIES} ${GTEST_INCLUDE_DIR})
message(STATUS "OpenCL headers:" ${OPENCL_INCLUDE_DIR})

include_directories(${GTEST_INCLUDE_DIR})
include_directories(${Boost_INCLUDE_DIR})
include_directories(../library/include)
include_directories(resources)

set(TESTS
test_hdl_list
test_clsparse_init
test_kernel_simple
test_csrmv_simple
)

#additional headers and sources
set(src
resources/csr_matrix_environment.cpp
)
set(headers
resources/hdl_test_resources.h
resources/opencl_utils.h
resources/clsparse_environment.h
resources/matrix_utils.h
resources/matrix_market.h
resources/csr_matrix_environment.h
)

foreach(test ${TESTS})
        add_executable(${test} ${test}.cpp ${src} ${headers})
        target_link_libraries(${test} clSPARSE)
        target_link_libraries(${test} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
        target_link_libraries(${test} ${Boost_LIBRARIES})
endforeach(test)
