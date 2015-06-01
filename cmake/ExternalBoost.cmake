message( STATUS "Configuring Boost SuperBuild..." )
include( ExternalProject )

# TODO:  Options should be added to allow downloading Boost straight from github

# This file is used to add Boost as a library dependency to another project
# This sets up boost to download from sourceforge, and builds it as a cmake
# ExternalProject

# Change this one line to upgrade to newer versions of boost
set( ext.Boost_VERSION "1.58.0" CACHE STRING "Boost version to download/use" )
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

set( Boost.Command ./b2 --prefix=<INSTALL_DIR>/package )

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
elseif( MSVC14 )
  list( APPEND Boost.Command toolset=msvc-14.0 )
endif( )

if( WIN32 )
  list( APPEND Boost.Command define=BOOST_LOG_USE_WINNT6_API )
endif( )

set( ext.Boost_LINK "static" CACHE STRING "Which boost link method?  static | shared | static,shared" )

if( WIN32 )
    # Versioned is the default on windows
    set( ext.Boost_LAYOUT "versioned" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

    # For windows, default to build both variants to support the VS IDE
    set( ext.Boost_VARIANT "debug,release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
else( )
    # Tagged builds provide unique enough names to be able to build both variants
    set( ext.Boost_LAYOUT "tagged" CACHE STRING "Which boost layout method?  versioned | tagged | system" )

   # For Linux, typically a build tree only needs one variant
   if( ${CMAKE_BUILD_TYPE} STREQUAL "Debug")
     set( ext.Boost_VARIANT "debug" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   else( )
     set( ext.Boost_VARIANT "release" CACHE STRING "Which boost variant?  debug | release | debug,release" )
   endif( )
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
    set( ext.MD5_HASH "b0605a9323f1e960f7434dbbd95a7a5c" )
  else( )
    # .7z file
    set( ext.MD5_HASH "f7255aeb692c1c38fe761c32fb0d3ecd" )
  endif( )
else( )
  set( Boost.Bootstrap "./bootstrap.sh" )

  # .tar.bz2
  set( ext.MD5_HASH "b8839650e61e9c1c0a89f371dd475546" )
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

set_property( TARGET Boost PROPERTY FOLDER "Externals" )
ExternalProject_Get_Property( Boost install_dir )

# For use by the user of ExternalBoost.cmake
set( BOOST_ROOT ${install_dir}/package )
