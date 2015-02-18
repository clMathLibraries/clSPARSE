project(tests)
cmake_minimum_required(VERSION 2.8)

find_package(GTest REQUIRED)

set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} -std=c++11)
message(STATUS "GTest libraries:" ${GTEST_LIBRARIES} ${GTEST_INCLUDE_DIR})
message(STATUS "OpenCL headers:" ${OPENCL_INCLUDE_DIR})

include_directories(${GTEST_INCLUDE_DIR})
include_directories(../library/include)
include_directories(resources)

set(TESTS
test_hdl_list
test_clsparse_init
)

#additional headers and sources
set(src

)
set(headers
resources/hdl_test_resources.h
)

foreach(test ${TESTS})
        add_executable(${test} ${test}.cpp ${src} ${headers})
        target_link_libraries(${test} clSPARSE)
        target_link_libraries(${test} ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES})
endforeach(test)
