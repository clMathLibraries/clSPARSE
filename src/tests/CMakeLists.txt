
find_package( GTest REQUIRED )

set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
set( Boost_NO_SYSTEM_PATHS ON )
set( Boost_ADDITIONAL_VERSIONS 1.57.0 1.57 )
# set( Boost_DEBUG ON )

find_package( Boost COMPONENTS program_options )

# Unit tests requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

add_definitions(-D__CL_ENABLE_EXCEPTIONS)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/library
  resources
  ${GTEST_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  )

set( TESTS_CPP
  test_clsparse_init
  test_kernel_simple
  test_csrmv_simple
  test_resources_io
  test_clsparse_transpose
  test_clsparse_csr2dense
  test_cl_traits
  test_readMMcoo
)

set( TESTS_C
  test_interface_c
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

foreach( test ${TESTS_CPP} )
  add_executable( ${test} ${test}.cpp ${src} ${headers} )
  target_link_libraries( ${test} clSPARSE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
   target_compile_options( ${test} PUBLIC -std=c++11 )
   if( CMAKE_COMPILER_IS_GNUCXX )
    target_link_libraries( ${test} pthread ) #gtest on linux brings this dependency
  endif( )
  endif( )
  set_target_properties( ${test} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${test} PROPERTY FOLDER "Tests" )
endforeach( )

foreach( test ${TESTS_C} )
  add_executable( ${test} ${test}.c ${headers} )
  target_link_libraries( ${test} clSPARSE ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
  if( CMAKE_COMPILER_IS_GNUCC )
    target_link_libraries( ${test} pthread ) #gtest on linux brings this dependency
  endif( )
  set_target_properties( ${test} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${test} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${test} PROPERTY FOLDER "Tests" )
endforeach( )
