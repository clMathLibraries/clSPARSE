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

  project( SuperBuild.clSPARSE VERSION 0.0.1.0 )
else( )
  project( SuperBuild.clSPARSE )

  # Define a version for the code
  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_MAJOR )
    set( SuperBuild.clSPARSE_VERSION_MAJOR 0 )
  endif( )

  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_MINOR )
    set( SuperBuild.clSPARSE_VERSION_MINOR 0 )
  endif( )

  if( NOT DEFINED SuperBuild.clSPARSE_VERSION_PATCH )
    set( SuperBuild.clSPARSE_VERSION_PATCH 1 )
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
option( BUILD_AppSdk "Install AMD AppSdk as an external dependency" ON )
option( BUILD_clBLAS "Install clBLAS as an external dependency" ON )

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
  list( APPEND clSPARSE.Cmake.Args -DGMOCK_ROOT=${GMOCK_ROOT} )
else( )
  if( NOT DEFINED GMOCK_ROOT AND NOT DEFINED ENV{GMOCK_ROOT} )
    message( SEND_ERROR "GMOCK_ROOT must be provided if BUILD_gMock is disabled" )
  else( )
    if( NOT DEFINED GMOCK_ROOT )
      set( GMOCK_ROOT "$ENV{GMOCK_ROOT}" )
    endif( )
    list( APPEND clSPARSE.Cmake.Args -DGMOCK_ROOT=${GMOCK_ROOT} )
  endif( )
endif( )

# If the user selects, download, uncompress, and setup clBLAS
if( BUILD_clBLAS )
  message( STATUS "Setting up clBLAS external..." )
  include( ExternalclBLAS )
  message( STATUS "CLMATH_BLAS_ROOT configured as: " ${CLMATH_BLAS_ROOT} )
  list( APPEND clSPARSE.Dependencies clMATH.clblas )
  list( APPEND clSPARSE.Cmake.Args -DCLMATH_BLAS_ROOT=${CLMATH_BLAS_ROOT} )
else( )
  if( NOT DEFINED CLMATH_BLAS_ROOT AND NOT DEFINED ENV{CLMATH_BLAS_ROOT} AND BUILD_clSPARSE )
    message( SEND_ERROR "CLMATH_BLAS_ROOT must be provided if BUILD_clBLAS is disabled" )
  else( )
    if( NOT DEFINED CLMATH_BLAS_ROOT )
      set( CLMATH_BLAS_ROOT "$ENV{CLMATH_BLAS_ROOT}" )
    endif( )
    list( APPEND clSPARSE.Cmake.Args -DCLMATH_BLAS_ROOT=${CLMATH_BLAS_ROOT} )
  endif( )
endif( )

# Main project
if( BUILD_clSPARSE )
  message( STATUS "Setting up clSPARSE external..." )
  include( ExternalProject )

  # Also add sparse as an external dependency.  The advantage of this is that sparse can be treated from cmake
  # as an independant project.  The developer can rebuild sparse project without rebuilding all dependencies.
  ExternalProject_Add(
    clSPARSE
    DEPENDS ${clSPARSE.Dependencies}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/src
    BINARY_DIR clSPARSE-build
    CMAKE_ARGS ${clSPARSE.Cmake.Args} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    INSTALL_COMMAND ""
  )
endif( )
