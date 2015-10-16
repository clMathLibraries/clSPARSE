# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
# Copyright 2015 Vratis, Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

message( STATUS "Configuring tests subdir" )

find_package( GTest REQUIRED )

# If user provides a specific boost path through command line, make sure to
# use that
if( DEFINED BOOST_ROOT OR DEFINED ENV{BOOST_ROOT} )
  set( Boost_NO_SYSTEM_PATHS ON )
  set( Boost_NO_BOOST_CMAKE ON )
endif( )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.59.0 1.59 1.58.0 1.58 1.57.0 1.57 )

# On windows, Boost prefers to link statically unless the user defines BOOST_ALL_DYN_LINK
# Therefore, we prefer linking  static libs first to avoid extra #defines
set( Boost_USE_STATIC_LIBS ON )
find_package( Boost COMPONENTS program_options serialization )

# If static libraries are not found, search again for shared libraries
if( NOT Boost_FOUND )
  set( Boost_USE_STATIC_LIBS OFF )
  find_package( Boost COMPONENTS program_options serialization )
endif( )

# Unit tests requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

#add_definitions(-D__CL_ENABLE_EXCEPTIONS)

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/library
  ${PROJECT_BINARY_DIR}/include
  resources
  #resources/uBLAS-linalg
  ${GTEST_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  )

set( TESTS_CPP
  test-blas1
  test-blas2
  test-blas3
  test-solvers
  test-conversion
  test-clsparse-utils
)

set( TESTS_C
  test_interface_c
  )

#additional headers and sources
set( src
  resources/csr_matrix_environment.cpp
  resources/opencl_utils.cpp
  resources/sparse_matrix_environment.cpp
)

set( headers
  resources/opencl_utils.h
  resources/clsparse_environment.h
  resources/matrix_utils.h
  resources/sparse_matrix_environment.h
  resources/csr_matrix_environment.h
  resources/blas1_environment.h
  resources/sparse_matrix_fill.hpp
  #resources/uBLAS-linalg/ublas_pcg.hpp
  #resources/uBLAS-linalg/precond.hpp
  #resources/uBLAS-linalg/cholesky.hpp
)

foreach( test ${TESTS_CPP} )
  add_executable( ${test} ${test}.cpp ${src} ${headers} )
  target_link_libraries( ${test} clSPARSE ${CMAKE_DL_LIBS} ${GTEST_LIBRARIES} ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
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
