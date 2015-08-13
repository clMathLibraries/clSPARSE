# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
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

message( STATUS "Configuring gMock SuperBuild..." )
include( ExternalProject )

set( ext.gMock_Version "1.7.0" CACHE STRING "gMock version to download/use" )
mark_as_advanced( ext.gMock_Version )

message( STATUS "ext.gMock_Version: " ${ext.gMock_Version} )

if( DEFINED ENV{GMOCK_URL} )
  set( ext.gMock_URL "$ENV{GMOCK_URL}" CACHE STRING "URL to download gMock from" )
else( )
  set( ext.gMock_URL "https://googlemock.googlecode.com/files/gmock-${ext.gMock_Version}.zip" CACHE STRING "URL to download gMock from" )
endif( )
mark_as_advanced( ext.gMock_URL )

# Create a workspace to house the src and buildfiles for googleMock
set_directory_properties( PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/Externals/gMock )

if( BUILD64 )
  set( LIB_DIR lib64 )
else( )
  set( LIB_DIR lib )
endif( )

set( ext.gMock.cmake_args -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${LIB_DIR} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package )

if( CMAKE_COMPILER_IS_GNUCC )
  if( BUILD64 )
    set( EXTRA_FLAGS "-m64" )
  else( )
    set( EXTRA_FLAGS "-m32" )
  endif( )

  list( APPEND ext.gMock.cmake_args -DCMAKE_C_FLAGS=${EXTRA_FLAGS} -DCMAKE_CXX_FLAGS=${EXTRA_FLAGS} )
endif( )

if( MSVC_IDE OR XCODE_VERSION )
  list( APPEND ext.gMock.cmake_args -Dgtest_force_shared_crt=ON )
  set( ext.gMock.Make
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
        COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Debug
  )
else( )
  # Add build thread in addition to the number of cores that we have
  include( ProcessorCount )
  ProcessorCount( Cores )
  message( STATUS "ExternalclBLAS detected ( " ${Cores} " ) cores to build clBLAS with" )

  set( ext.gMock.Make "make" )
  if( NOT Cores EQUAL 0 )
    math( EXPR Cores "${Cores} + 1 " )
    list( APPEND ext.gMock.Make -j ${Cores} )
  else( )
    # If we could not detect # of cores, assume 1 core and add an additional build thread
    list( APPEND ext.gMock.Make -j 2 )
  endif( )
endif( )

# Add external project for googleMock
ExternalProject_Add(
  gMock
  URL ${ext.gMock_URL}
  URL_MD5 073b984d8798ea1594f5e44d85b20d66
  CMAKE_ARGS ${ext.gMock.cmake_args} -DCMAKE_DEBUG_POSTFIX=d
  BUILD_COMMAND ${ext.gMock.Make}
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property( gMock source_dir )

# For visual studio, the path 'debug' is hardcoded because that is the default VS configuration for a build.
# Doesn't matter if its the gMock or gMockd project above
set( packageDir "<INSTALL_DIR>/package" )

set( gMockLibDir "<BINARY_DIR>/${LIB_DIR}" )
set( gTestLibDir "<BINARY_DIR>/gtest/${LIB_DIR}" )
if( MSVC_IDE OR XCODE_VERSION )
    # Create a package by bundling libraries and header files
    ExternalProject_Add_Step( gMock createPackage
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gMockLibDir}/Debug ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gMockLibDir}/Release ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gTestLibDir}/Debug ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gTestLibDir}/Release ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${packageDir}/include
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest ${packageDir}/include/gtest
      DEPENDEES install
    )
else( )
    # Create a package by bundling libraries and header files
    ExternalProject_Add_Step( gMock createPackage
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gMockLibDir} ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory ${gTestLibDir} ${packageDir}/${LIB_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${packageDir}/include
      COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest ${packageDir}/include/gtest
      DEPENDEES install
    )
endif( )

set_property( TARGET gMock PROPERTY FOLDER "Externals")
ExternalProject_Get_Property( gMock install_dir )

# For use by the user of ExternalGtest.cmake
set( GMOCK_ROOT ${install_dir}/package )
