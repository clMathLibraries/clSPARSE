project(tests)
cmake_minimum_required(VERSION 2.8)

find_package(GTest REQUIRED)

set( Boost_USE_MULTITHREADED ON )
if (WIN32)
    set( Boost_USE_STATIC_LIBS   ON )
else()
    set( Boost_USE_STATIC_LIBS   ON )
endif()
set( Boost_DETAILED_FAILURE_MSG   ON )

# set( Boost_DEBUG ON )
find_package(Boost COMPONENTS
    program_options
)

if( NOT WIN32 )
    set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} -std=c++11)
endif( )

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
test_validation
test_clsparse_spmv_ctrl
test_resources_io
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
