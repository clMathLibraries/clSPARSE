message( STATUS "Configuring gMock SuperBuild..." )
include( ExternalProject )

set( ext.gMock_Version "1.7.0" CACHE STRING "gMock version to download/use" )
mark_as_advanced( ext.gMock_Version )

message( STATUS "ext.gMock_Version: " ${ext.gMock_Version} )

if( BUILD64 )
  set( LIB_DIR lib64 )
  set( LIB_FLAG "-m64" )
else( )
  set( LIB_DIR lib )
  set( LIB_FLAG "-m32" )
endif( )

if( DEFINED ENV{GMOCK_URL} )
  set( ext.gMock_URL "$ENV{GMOCK_URL}" CACHE STRING "URL to download gMock from" )
else( )
  set( ext.gMock_URL "https://googlemock.googlecode.com/files/gmock-${ext.gMock_Version}.zip" CACHE STRING "URL to download gMock from" )
endif( )
mark_as_advanced( ext.gMock_URL )

# Create a workspace to house the src and buildfiles for googleMock
set_directory_properties( PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/Externals/gMock )

# FindGTest.cmake assumes that debug gtest libraries end with a 'd' postfix.  The official gtest cmakelist files do not add this postfix,
# but luckily cmake allows us to specify a postfix through the CMAKE_DEBUG_POSTFIX variable.

set( ext.gMock.cmake_args -DCMAKE_ARCHIVE_OUTPUT_DIRECTORY:PATH=${LIB_DIR} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DCMAKE_C_FLAGS=${LIB_FLAG} -DCMAKE_CXX_FLAGS=${LIB_FLAG} )
if( WIN32 )
  list( APPEND ext.gMock.cmake_args -Dgtest_force_shared_crt=ON -DCMAKE_DEBUG_POSTFIX=d )
endif( )

# Add external project for googleMock
ExternalProject_Add(
  gMock
  URL ${ext.gMock_URL}
  URL_MD5 073b984d8798ea1594f5e44d85b20d66
  PATCH_COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gMock.vs11.patch.txt" <SOURCE_DIR>/CMakeLists.txt
  CMAKE_ARGS ${ext.gMock.cmake_args}
)

# Need to patch the googletest cmakefile
ExternalProject_Add_Step( gMock gTestPatch
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gTest.vs11.patch.txt" <SOURCE_DIR>/gtest/CMakeLists.txt
  DEPENDEES download
  DEPENDERS  patch
)

# Need to copy the header files to the staging directory too
ExternalProject_Add_Step( gMock copyHeaders
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/include <SOURCE_DIR>/../../package/include
  COMMAND ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/gtest/include/gtest <SOURCE_DIR>/../../package/include/gtest
  DEPENDEES install
)

set_property( TARGET gMock PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( gMock source_dir )

# For use by the user of ExternalGtest.cmake
set( GMOCK_FOUND TRUE )
set( GMOCK_ROOT ${source_dir}/../../package )
