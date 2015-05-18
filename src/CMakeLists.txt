# We require 2.8.12 for linking libraries we specify PRIVATE, INTERFACE and
# PUBLIC keywords
cmake_minimum_required( VERSION 2.8.12 )

if( CMAKE_GENERATOR MATCHES "NMake" )
	option( NMAKE_COMPILE_VERBOSE "Print VERBOSE compile/link msgs to the console" OFF )
	if( NMAKE_COMPILE_VERBOSE )
		set( CMAKE_START_TEMP_FILE "" )
		set( CMAKE_END_TEMP_FILE "" )
		set( CMAKE_VERBOSE_MAKEFILE 1 )
	endif( )
endif( )

# Check if cmake supports the new VERSION tag for project() commands
# clSPARSE becomes the name of the project with a particular version
if( POLICY CMP0048 )
	cmake_policy( SET CMP0048 NEW )
	project( clSPARSE VERSION 0.0.1.0 LANGUAGES C CXX )
else( )
	project( clSPARSE C CXX )
	# Define a version for the code
	if( NOT DEFINED clSPARSE_VERSION_MAJOR )
		set( clSPARSE_VERSION_MAJOR 0 )
	endif( )

	if( NOT DEFINED clSPARSE_VERSION_MINOR )
		set( clSPARSE_VERSION_MINOR 0 )
	endif( )

	if( NOT DEFINED clSPARSE_VERSION_PATCH )
		set( clSPARSE_VERSION_PATCH 1 )
	endif( )

	if( NOT DEFINED clSPARSE_VERSION_TWEAK )
		set( clSPARSE_VERSION_TWEAK 0 )
	endif( )

	set( clSPARSE_VERSION "${clSPARSE_VERSION_MAJOR}.${clSPARSE_VERSION_MINOR}.${clSPARSE_VERSION_PATCH}.${clSPARSE_VERSION_TWEAK}")
endif( )

message( STATUS "clSPARSE_VERSION_MAJOR=${clSPARSE_VERSION_MAJOR}" )
message( STATUS "clSPARSE_VERSION_MINOR=${clSPARSE_VERSION_MINOR}" )
message( STATUS "clSPARSE_VERSION_PATCH=${clSPARSE_VERSION_PATCH}" )
message( STATUS "clSPARSE_VERSION_TWEAK=${clSPARSE_VERSION_TWEAK}" )

# This is incremented when the ABI to the library changes
set( clSPARSE_SOVERSION 1 )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake )

# On windows, it's convenient to change the default install prefix such that it does NOT point to 'program files'
# Need to check out CMAKE_RUNTIME_OUTPUT_DIRECTORY variable, and see if that eliminates the need to modify install path
if( WIN32 AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT )
	set( CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories" FORCE )
endif( )

if( MSVC_IDE )
	set( clSPARSE_BUILD64 ${CMAKE_CL_64} )
	set_property( GLOBAL PROPERTY USE_FOLDERS TRUE )
else()
	option( clSPARSE_BUILD64 "Build a 64-bit product" ON )
endif()

# These variables are meant to contain string which should be appended to the installation paths
# of library and executable binaries, respectively.  They are meant to be user configurable/overridable.
set( SUFFIX_LIB_DEFAULT "" )
set( SUFFIX_BIN_DEFAULT "" )

# Modify the global find property to help us find libraries like Boost in the correct paths for 64-bit
# Essentially, find_library calls will look for /lib64 instead of /lib; works for windows and linux
if( clSPARSE_BUILD64 )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
	message( STATUS "64bit build - FIND_LIBRARY_USE_LIB64_PATHS TRUE" )

	if( NOT APPLE )
		set( SUFFIX_LIB_DEFAULT "64" )
	endif( )
else( )
	set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS FALSE )
	message( STATUS "32bit build - FIND_LIBRARY_USE_LIB64_PATHS FALSE" )
endif( )

set( SUFFIX_LIB ${SUFFIX_LIB_DEFAULT} CACHE STRING "String to append to 'lib' install path" )
set( SUFFIX_BIN ${SUFFIX_BIN_DEFAULT} CACHE STRING "String to append to 'bin' install path" )

# Currently, linux has a problem outputing both narrow and wide characters,
# which happens in our client because openCL only supports narrow characters
if( WIN32 )
	option( BUILD_UNICODE "Create a solution that compiles clSPARSE with Unicode Support" ON )
	if( BUILD_UNICODE )
		message( STATUS "UNICODE build" )
	endif( )
else()
	set( BUILD_UNICODE OFF )
	message( STATUS "UNICODE disabled on linux" )
endif()

# avoid warning on newer cmake versions
if( APPLE )
  set( CMAKE_MACOSX_RPATH 0 )
endif( )

# Set common compile and link options
if( MSVC )
	# Following options for nMake
	message( STATUS "Detected MSVS Ver: " ${MSVC_VERSION} )

	# CMake uses huge stack frames for windows, for some reason.  We remove.
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS}" )
	string( REGEX REPLACE "/STACK:[0-9]+" "" CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS}" )

	if( BUILD_StripSymbols )
		string( REGEX REPLACE "/Zi" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}" )
	endif()
elseif( CMAKE_COMPILER_IS_GNUCC )
	message( STATUS "Detected GNU compiler collection" )

	if( clSPARSE_BUILD64 )
		set( CMAKE_CXX_FLAGS "-m64 ${CMAKE_CXX_FLAGS}" )
		set( CMAKE_C_FLAGS "-m64 ${CMAKE_C_FLAGS}" )
	else( )
		set( CMAKE_CXX_FLAGS "-m32 ${CMAKE_CXX_FLAGS}" )
		set( CMAKE_C_FLAGS "-m32 ${CMAKE_C_FLAGS}" )
	endif( )
else( )
	message( AUTHOR_WARNING "Compiler not recognized.  Using default flags." )
endif( )

# If UNICODE is defined, pass extra definitions into
if( BUILD_UNICODE )
	add_definitions( "/DUNICODE /D_UNICODE" )
endif( )

# Print out compiler flags for viewing/debug
message( STATUS "CMAKE_CXX_COMPILER flags: " ${CMAKE_CXX_FLAGS} )
message( STATUS "CMAKE_CXX_COMPILER debug flags: " ${CMAKE_CXX_FLAGS_DEBUG} )
message( STATUS "CMAKE_CXX_COMPILER release flags: " ${CMAKE_CXX_FLAGS_RELEASE} )
message( STATUS "CMAKE_CXX_COMPILER relwithdebinfo flags: " ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} )
message( STATUS "CMAKE_EXE_LINKER link flags: " ${CMAKE_EXE_LINKER_FLAGS} )

# configure a header file to pass the CMake version settings to the source, and package the header files in the output archive
message( STATUS "PROJECT SRC DIR = ${PROJECT_SOURCE_DIR}")
configure_file( "${PROJECT_SOURCE_DIR}/include/clSPARSE.version.h.in" "${PROJECT_BINARY_DIR}/include/clSPARSE.version.h" )
install( FILES
			"${PROJECT_BINARY_DIR}/include/clSPARSE.version.h"
			"include/clSPARSE.h"
			"include/clSPARSE_1x.h"
			"include/clSPARSE_2x.h"
		DESTINATION
			"./include" )

option( BUILD_TESTS "Build clSPARSE unit tests; requires Boost & googletest" ON )
option( BUILD_BENCHMARKS "Build clSPARSE benchmarks" ON )

add_subdirectory( library )
add_subdirectory( clsparseTimer )

if( BUILD_BENCHMARKS )
	add_subdirectory( benchmarks )
endif( )

if( BUILD_TESTS )
	add_subdirectory( tests )
endif( )

# Depending on whether we are building for 64 or 32 bit, construct common paths and names that subdirectories can reference for their use
if( clSPARSE_BUILD64 )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${clSPARSE_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x64" )
	set( INCLUDE_DIR include )
else( )
	set( CPACK_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${clSPARSE_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-x32" )
	set( INCLUDE_DIR include )
endif( )

# The following code is setting variables to control the behavior of CPack to generate our
if( WIN32 )
	set( CPACK_SOURCE_GENERATOR "ZIP" )
	set( CPACK_GENERATOR "ZIP" )
else( )
	set( CPACK_SOURCE_GENERATOR "TGZ" )
	set( CPACK_GENERATOR "TGZ" )
endif( )

set( CPACK_SOURCE_PACKAGE_FILE_NAME "${CMAKE_PROJECT_NAME}-${clSPARSE_VERSION}-${CMAKE_HOST_SYSTEM_NAME}-Source")
# set( CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt")
set( CPACK_PACKAGE_VERSION_MAJOR ${clSPARSE_VERSION_MAJOR} )
set( CPACK_PACKAGE_VERSION_MINOR ${clSPARSE_VERSION_MINOR} )
set( CPACK_PACKAGE_VERSION_PATCH ${clSPARSE_VERSION_PATCH} )
set( CPACK_PACKAGE_VERSION_TWEAK ${clSPARSE_VERSION_TWEAK} )
set( CPACK_PACKAGE_DESCRIPTION_SUMMARY "OpenCL SPARSE library package")
set( CPACK_PACKAGE_VENDOR "AMD")
set( CPACK_SOURCE_IGNORE_FILES "/\\\\.git/;/\\\\.hg/;/\\\\.svn/;" )

# Define all variables that influence CPack before including CPack, such as install targets
include( CPack )
