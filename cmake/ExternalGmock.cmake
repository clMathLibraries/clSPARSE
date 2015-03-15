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
    set( EXTRA_FLAGS "-m64 -pthread" )
  else( )
    set( EXTRA_FLAGS "-m32 -pthread" )
  endif( )

  list( APPEND ext.gMock.cmake_args -DCMAKE_C_FLAGS=${EXTRA_FLAGS} -DCMAKE_CXX_FLAGS=${EXTRA_FLAGS} )
endif( )

if( MSVC )
  list( APPEND ext.gMock.cmake_args -Dgtest_force_shared_crt=ON )
endif( )

# Add external project for googleMock
ExternalProject_Add(
  gMock
  URL ${ext.gMock_URL}
  URL_MD5 073b984d8798ea1594f5e44d85b20d66
  CMAKE_ARGS ${ext.gMock.cmake_args}
  INSTALL_COMMAND ""
)

ExternalProject_Get_Property( gMock source_dir )

# FindGTest.cmake assumes that debug gtest libraries end with a 'd' postfix.  The official gtest cmakelist files do not add this postfix,
# but luckily cmake allows us to specify a postfix through the CMAKE_DEBUG_POSTFIX variable.
ExternalProject_Add(
  gMockd
  DEPENDS gMock
  URL ${source_dir}
  CMAKE_ARGS ${ext.gMock.cmake_args} -DCMAKE_DEBUG_POSTFIX=d
  INSTALL_COMMAND ""
)

# For visual studio, the path 'debug' is hardcoded because that is the default VS configuration for a build.
# Doesn't matter if its the gMock or gMockd project above
if( MSVC )
  set( gMockLibDir "<BINARY_DIR>/${LIB_DIR}/Debug" )
  set( gTestLibDir "<BINARY_DIR>/gtest/${LIB_DIR}/Debug" )
else( )
  set( gMockLibDir "<BINARY_DIR>/${LIB_DIR}" )
  set( gTestLibDir "<BINARY_DIR>/gtest/${LIB_DIR}" )
endif( )

set( packageDir "<SOURCE_DIR>/../../package" )

# Create a package by bundling libraries and header files
ExternalProject_Add_Step( gMock createPackage
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${gMockLibDir} ${packageDir}/${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${gTestLibDir} ${packageDir}/${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include ${packageDir}/include
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest ${packageDir}/include/gtest
  DEPENDEES install
)

# Header file are the same and can be excluded
ExternalProject_Add_Step( gMockd createPackage
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${gMockLibDir} ${packageDir}/${LIB_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${gTestLibDir} ${packageDir}/${LIB_DIR}
  DEPENDEES install
)

set_property( TARGET gMock PROPERTY FOLDER "Externals")
set_property( TARGET gMockd PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( gMock install_dir )

# For use by the user of ExternalGtest.cmake
set( GMOCK_FOUND TRUE )
set( GMOCK_ROOT ${install_dir}/package )
