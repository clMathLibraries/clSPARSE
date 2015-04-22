message( STATUS "Configuring Boost SuperBuild..." )
include( ExternalProject )

set( ext.Boost_VERSION "1.57.0" CACHE STRING "Boost version to download/use" )
mark_as_advanced( ext.Boost_VERSION )
string( REPLACE "." "_" ext.Boost_Version_Underscore ${ext.Boost_VERSION} )

message( STATUS "ext.Boost_VERSION: " ${ext.Boost_VERSION} )

if( WIN32 )
  # For newer cmake versions, 7z archives are much smaller to download
  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    set( Boost_Ext "zip" )
  else( )
    set( Boost_Ext "7z" )
  endif( )
else( )
  set( Boost_Ext "tar.bz2" )
endif( )

# Purely for debugging the file downloading URLs
# file( DOWNLOAD "http://downloads.sourceforge.net/project/boost/boost/1.55.0/boost_1_55_0.7z"
# "${CMAKE_CURRENT_BINARY_DIR}/download/boost-${ext.Boost_VERSION}/boost_1_55_0.7z" SHOW_PROGRESS STATUS fileStatus LOG fileLog )
# message( STATUS "status: " ${fileStatus} )
# message( STATUS "log: " ${fileLog} )

set( Boost.Command ./b2 --prefix=<SOURCE_DIR>/../package )

# message( "GNUCC: ${CMAKE_COMPILER_IS_GNUCC}" )
# message( "GNUCXX: ${CMAKE_COMPILER_IS_GNUCXX}" )
if( CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX )
  list( APPEND Boost.Command cxxflags=-fPIC  )
endif( )

include( ProcessorCount )
ProcessorCount( Cores )
message( STATUS "ExternalBoost detected ( " ${Cores} " ) cores to build Boost with" )
if( NOT Cores EQUAL 0 )
  # Add build thread in addition to the number of cores that we have
  math( EXPR Cores "${Cores} + 1 " )
else( )
  # If we could not detect # of cores, assume 1 core and add an additional build thread
  set( Cores "2" )
endif( )
list( APPEND Boost.Command -j ${Cores} --with-program_options --with-filesystem --with-system --with-regex )

if( BUILD64 )
  list( APPEND Boost.Command address-model=64 )
else( )
  list( APPEND Boost.Command address-model=32 )
endif( )

if( MSVC10 )
  list( APPEND Boost.Command toolset=msvc-10.0 )
elseif( MSVC11 )
  list( APPEND Boost.Command toolset=msvc-11.0 )
elseif( MSVC12 )
  list( APPEND Boost.Command toolset=msvc-12.0 )
endif( )

if( WIN32 )
  list( APPEND Boost.Command define=BOOST_LOG_USE_WINNT6_API )
endif( )

set( ext.Boost_VARIANT "debug,release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
set( ext.Boost_LINK "static" CACHE STRING "Which boost link method?  static | shared | static,shared" )

if( WIN32 )
   set( ext.Boost_LAYOUT "versioned" CACHE STRING "Which boost layout method?  versioned | tagged | system" )
else( )
   set( ext.Boost_LAYOUT "tagged" CACHE STRING "Which boost layout method?  versioned | tagged | system" )
endif( )

list( APPEND Boost.Command link=${ext.Boost_LINK} variant=${ext.Boost_VARIANT} --layout=${ext.Boost_LAYOUT} install )

message( STATUS "Boost.Command: ${Boost.Command}" )

# If the user has a cached local copy stored somewhere, they can define the full path to the package in a BOOST_URL environment variable
if( DEFINED ENV{BOOST_URL} )
  set( ext.Boost_URL "$ENV{BOOST_URL}" CACHE STRING "URL to download Boost from" )
else( )
  set( ext.Boost_URL "http://sourceforge.net/projects/boost/files/boost/${ext.Boost_VERSION}/boost_${ext.Boost_Version_Underscore}.${Boost_Ext}/download" CACHE STRING "URL to download Boost from" )
endif( )
mark_as_advanced( ext.Boost_URL )

set( Boost.Bootstrap "" )
set( ext.MD5_HASH "" )
if( WIN32 )
  set( Boost.Bootstrap "./bootstrap.bat" )

  if( CMAKE_VERSION VERSION_LESS "3.1.0" )
    # .zip file
    set( ext.MD5_HASH "5e040e578e3f0ba879da04a1e0cd55ff" )
  else( )
    # .7z file
    set( ext.MD5_HASH "17c98dd78d6180f553fbefe5a0f57d12" )
  endif( )
else( )
  set( Boost.Bootstrap "./bootstrap.sh" )

  # .tar.bz2
  set( ext.MD5_HASH "1be49befbdd9a5ce9def2983ba3e7b76" )
endif( )

# Below is a fancy CMake command to download, build and install Boost on the users computer
ExternalProject_Add(
  Boost
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/Externals/Boost
  URL ${ext.Boost_URL}
  URL_MD5 ${ext.MD5_HASH}
  UPDATE_COMMAND ${Boost.Bootstrap}
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ${Boost.Command}
  BUILD_IN_SOURCE 1
  INSTALL_COMMAND ""
)

set_property( TARGET Boost PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( Boost source_dir )

set( Boost_FOUND TRUE )
set( BOOST_ROOT ${source_dir}/../package )
