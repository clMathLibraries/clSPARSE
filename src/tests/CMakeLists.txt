

message( STATUS "GTEST_ROOT: " ${GTEST_ROOT} )
find_package( GTest REQUIRED )

set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
# set( Boost_DEBUG ON )

find_package( Boost COMPONENTS program_options )

# Unit tests requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

if( CMAKE_COMPILER_IS_GNUCC )
  set( CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} )
endif( )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/library
  resources
  ${GTEST_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  )

set( TESTS
#  test_hdl_list
  test_clsparse_init
  test_kernel_simple
  test_csrmv_simple
#  test_validation
  test_clsparse_spmv_ctrl
  test_resources_io
  test_clsparse_transpose
)

#additional headers and sources
set( src
  resources/csr_matrix_environment.cpp
)

set( headers
  resources/hdl_test_resources.h
  resources/opencl_utils.h
  resources/clsparse_environment.h
  resources/matrix_utils.h
  resources/matrix_market.h
  resources/csr_matrix_environment.h
)

foreach( test ${TESTS} )
  add_executable( ${test} ${test}.cpp ${src} ${headers} )
  target_link_libraries( ${test} clSPARSE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
  set_property( TARGET ${test} PROPERTY FOLDER "Tests")
endforeach( )
