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

# We require 2.8.10 because of the added support to download from https URL's
cmake_minimum_required( VERSION 2.8.10 )

# uncomment these to debug nmake and borland makefiles
#SET(CMAKE_START_TEMP_FILE "")
#SET(CMAKE_END_TEMP_FILE "")
#SET(CMAKE_VERBOSE_MAKEFILE 1)

# Check if cmake supports the new VERSION tag for project() commands
# SuperBuild.clSPARSE becomes the name of the project with a particular version
if( POLICY CMP0048 )
  cmake_policy( SET CMP0048 NEW )

  project( SuperBuild.clSPARSE VERSION 0.7.0.0 )
else( )
  project( SuperBuild.clSPARSE )

  # Define a version for the code
  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_MAJOR )
    set( SuperBuild.clSPARSE_VERSION_MAJOR 0 )
  endif( )

  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_MINOR )
    set( SuperBuild.clSPARSE_VERSION_MINOR 7 )
  endif( )

  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_PATCH )
    set( SuperBuild.clSPARSE_VERSION_PATCH 0 )
  endif( )

  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_TWEAK )
    set( SuperBuild.clSPARSE_VERSION_TWEAK 0 )
  endif( )
endif( )

set( SuperBuild.clSPARSE_VERSION "${SuperBuild.clSPARSE_VERSION_MAJOR}.${SuperBuild.clSPARSE_VERSION_MINOR}.${SuperBuild.clSPARSE_VERSION_PATCH}")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# Set the default of CMAKE_BUILD_TYPE to be release, unless user specifies with -D.  MSVC_IDE does not use CMAKE_BUILD_TYPE
if( NOT MSVC_IDE AND NOT CMAKE_BUILD_TYPE )
  set( CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." FORCE )
endif()

if( MSVC_IDE )
  set( BUILD64 ${CMAKE_CL_64} )
  set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else( )
  option( BUILD64 "Build a 64-bit product" ON )
endif( )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( BUILD64 )
  set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
  message( STATUS "64bit build - FIND_LIBRARY_USE_LIB64_PATHS TRUE" )
else( )
  set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
  message( STATUS "32bit build - FIND_LIBRARY_USE_LIB64_PATHS FALSE" )
endif( )

# Various options below on configuring the build, and how to generate the solution files
option( BUILD_clSPARSE "Setup clSPARSE to use all the external dependencies" ON )
option( BUILD_Boost "Install Boost as an external dependency" ON )
option( BUILD_gMock "Install googleMock as an external dependency" ON )
#option( BUILD_clBLAS "Install clBLAS as an external dependency" ON )
option( BUILD_MTX "Download sparse matrix files for bench/test" OFF )
option( BUILD_SAMPLES "Build sample program using clSPARSE lib" OFF )
# Query the user for which version of OpenCL they wish to build the library for
set( BUILD_CLVERSION "1.2" CACHE STRING "The version of OpenCL we wish to compile the library against" )
set_property( CACHE BUILD_CLVERSION PROPERTY STRINGS 2.0 1.2 1.1 )


# Comment this out because this does not work yet
set( clSPARSE.Dependencies )
set( clSPARSE.Cmake.Args )

# If the user selects, download, uncompress, and setup Boost
if( BUILD_Boost )
  message( STATUS "Setting up Boost external..." )
  include( ExternalBoost )
  message( STATUS "BOOST_ROOT configured as: " ${BOOST_ROOT} )
  list( APPEND clSPARSE.Dependencies Boost )
  list( APPEND clSPARSE.Cmake.Args -DBOOST_ROOT=${BOOST_ROOT} )
else( )
  if( NOT DEFINED BOOST_ROOT AND NOT DEFINED ENV{BOOST_ROOT} )
    message( SEND_ERROR "BOOST_ROOT must be provided if BUILD_Boost is disabled" )
  else( )
    if( NOT DEFINED BOOST_ROOT )
      set( BOOST_ROOT "$ENV{BOOST_ROOT}" )
    endif( )
    list( APPEND clSPARSE.Cmake.Args -DBOOST_ROOT=${BOOST_ROOT} )
  endif( )
endif( )

  # If the user selects, download, uncompress, and setup googleMock
if( BUILD_gMock )
  message( STATUS "Setting up googleMock external..." )
  include( ExternalGmock )
  message( STATUS "GMOCK_ROOT configured as: " ${GMOCK_ROOT} )
  # list( APPEND clSPARSE.Dependencies gMockDebug gMockRelease )
  list( APPEND clSPARSE.Dependencies gMock )
  list( APPEND clSPARSE.Cmake.Args -DGTEST_ROOT=${GMOCK_ROOT} )
else( )
  if( NOT DEFINED GMOCK_ROOT AND NOT DEFINED ENV{GMOCK_ROOT} )
    message( SEND_ERROR "GMOCK_ROOT must be provided if BUILD_gMock is disabled" )
  else( )
    if( NOT DEFINED GMOCK_ROOT )
      set( GMOCK_ROOT "$ENV{GMOCK_ROOT}" )
    endif( )
    list( APPEND clSPARSE.Cmake.Args -DGTEST_ROOT=${GMOCK_ROOT} )
  endif( )
endif( )

# If the user selects, download, uncompress, and setup clBLAS
#if( BUILD_clBLAS )
#  message( STATUS "Setting up clBLAS external..." )
#  include( ExternalclBLAS )
#  message( STATUS "CLMATH_BLAS_ROOT configured as: " ${CLMATH_BLAS_ROOT} )
#  list( APPEND clSPARSE.Dependencies clMATH.clblas )
#  list( APPEND clSPARSE.Cmake.Args -DCLMATH_BLAS_ROOT=${CLMATH_BLAS_ROOT} )
#else( )
#  if( NOT DEFINED CLMATH_BLAS_ROOT AND NOT DEFINED ENV{CLMATH_BLAS_ROOT} AND BUILD_clSPARSE )
#    message( SEND_ERROR "CLMATH_BLAS_ROOT must be provided if BUILD_clBLAS is disabled" )
#  else( )
#    if( NOT DEFINED CLMATH_BLAS_ROOT )
#      set( CLMATH_BLAS_ROOT "$ENV{CLMATH_BLAS_ROOT}" )
#    endif( )
#    list( APPEND clSPARSE.Cmake.Args -DCLMATH_BLAS_ROOT=${CLMATH_BLAS_ROOT} )
#  endif( )
#endif( )

# If the user selects, download, uncompress, and setup clBLAS
if( BUILD_MTX )
  message( STATUS "Setting up Matrix Market data external..." )
  include( ExternalMTX )
endif( )

if( BUILD_clSPARSE OR BUILD_SAMPLES )
  include( ExternalProject )

  # Main project
  if( BUILD_clSPARSE )
    message( STATUS "Setting up clSPARSE external..." )

    if ( BUILD_SAMPLES )
      # If the user elects to build samples, then we must build and install clSPARSE
      # The install location is set as a subdirectory of the SuperBuild
      ExternalProject_Add(
        clSPARSE
        DEPENDS ${clSPARSE.Dependencies}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
        BINARY_DIR clSPARSE-build
        INSTALL_DIR clSPARSE-package
        CMAKE_ARGS ${clSPARSE.Cmake.Args} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_CLVERSION=${BUILD_CLVERSION} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
      )
    else( )
      # If the user does not want to build samples, then build clSPARSE but don't automatically
      # install it
      ExternalProject_Add(
        clSPARSE
        DEPENDS ${clSPARSE.Dependencies}
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
        BINARY_DIR clSPARSE-build
        CMAKE_ARGS ${clSPARSE.Cmake.Args} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_CLVERSION=${BUILD_CLVERSION}
        INSTALL_COMMAND ""
      )
    endif( )

  endif( )

  # Our goal with samples is to demonstrate two things
  # 1.  Demonstrate what simple clSPARSE code looks like to set up and execute common functionality
  # 2.  Demonstrate how an external program would find and link clSPARSE
  if ( BUILD_SAMPLES )
    message( STATUS "Setting up sample programs ...")
    ExternalProject_Get_Property( clSPARSE install_dir )

    ExternalProject_Add(
      clSPARSE-samples
      DEPENDS clSPARSE
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/samples
      BINARY_DIR clSPARSE-samples-build
      CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DBUILD_CLVERSION=${BUILD_CLVERSION} -DBUILD64=${BUILD64} -DCMAKE_PREFIX_PATH=${install_dir}
      EXCLUDE_FROM_ALL 1
      INSTALL_COMMAND ""
    )
  endif( )
endif( )
