# ########################################################################
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

cmake_minimum_required( VERSION 2.8.12 )
project(samples)

message( STATUS "BUILDING EXAMPLES")
message( STATUS "BUILD64: ${BUILD64}")

set( CLSPARSE_LIB_SUFFIX "")
if( ${BUILD64} )
   set( CLSPARSE_LIB_SUFFIX "64")
   set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
endif( )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find clSPARSE library
find_package( clSPARSE REQUIRED CONFIG )
message( STATUS "clSPARSE INCLUDES = ${clSPARSE_INCLUDE_DIR}")

# Query the user for which version of OpenCL they wish to build the library for
set( BUILD_CLVERSION "1.2" CACHE STRING "The version of OpenCL we wish to compile the samples against" )
set_property( CACHE BUILD_CLVERSION PROPERTY STRINGS 2.0 1.2 1.1 )
message( STATUS "clSPARSE samples are building using CL interface ='${BUILD_CLVERSION}'" )

# Find OpenCL
find_package( OpenCL ${BUILD_CLVERSION} REQUIRED )
message( STATUS "OPENCL_INCLUDE_DIRS = ${OPENCL_INCLUDE_DIRS}")

if( BUILD_CLVERSION VERSION_EQUAL "2.0" )
    add_definitions( -DBUILD_CLVERSION=200 )
elseif( BUILD_CLVERSION VERSION_EQUAL "1.2" )
    add_definitions( -DBUILD_CLVERSION=120 )
elseif( BUILD_CLVERSION VERSION_EQUAL "1.1" )
    add_definitions( -DBUILD_CLVERSION=110 )
endif( )

if(not USE_SYSTEM_CL2HPP)
    include(BuildCL2HPP)
endif(not USE_SYSTEM_CL2HPP)

include_directories(
    ${OPENCL_INCLUDE_DIRS}
    ${CL2HPP_INCLUDE_DIRECTORY}
    )

# No we are ready to generate the samples

set (SAMPLES_CPP
sample-initialization
sample-axpy
sample-spmv
sample-cg
)

set (SAMPLES_C
sample-initialization-c
sample-norm1-c
)

set(EXAMPLE_TARGETS)

foreach( sample ${SAMPLES_CPP} )
  add_executable( ${sample} ${sample}.cpp)
  target_link_libraries( ${sample} clSPARSE ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} )
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
   target_compile_options( ${sample} PUBLIC -std=c++11 )
  endif( )
  set_target_properties( ${sample} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${sample} PROPERTY FOLDER "Samples" )
  # collect targets for installation
  list(APPEND EXAMPLE_TARGETS ${sample})
  if(not USE_SYSTEM_CL2HPP)
      add_dependencies(${sample} cl2hpp)
  endif(not USE_SYSTEM_CL2HPP)
endforeach( )

foreach( sample ${SAMPLES_C} )
  add_executable( ${sample} ${sample}.c)
  target_link_libraries( ${sample} clSPARSE ${OPENCL_LIBRARIES})
  set_target_properties( ${sample} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${sample} PROPERTY FOLDER "Samples" )
# collect targets for installation
  list(APPEND EXAMPLE_TARGETS ${sample})
  if(not USE_SYSTEM_CL2HPP)
      add_dependencies(${sample} cl2hpp)
  endif(not USE_SYSTEM_CL2HPP)
endforeach( )

#Installation step
message(STATUS "DEST ${CMAKE_PREFIX_PATH}/Samples")
message(STATUS "TARGETS ${EXAMPLE_TARGETS}")
install(TARGETS ${EXAMPLE_TARGETS} DESTINATION  ${CMAKE_PREFIX_PATH}/Samples)
