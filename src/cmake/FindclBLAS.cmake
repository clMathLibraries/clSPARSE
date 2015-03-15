
# Locate clBLAS installation
#
# Defines the following variables:
#
#   CLBLAS_FOUND - Boolean indicating that the clBLAS installation was found without problems
#   clBLAS_INCLUDE_DIRS - Include directories
#
# Also defines the library variables below as normal
# variables.  These contain debug/optimized keywords when
# a debugging library is found.
#
#   clBLAS_LIBRARIES - libclBLAS
#
# Accepts the following variables as input:
#
#   CLMATH_BLAS_ROOT - (as a CMake or environment variable)
#                The root directory of the clMath implementation found
#
#   FIND_LIBRARY_USE_LIB64_PATHS - Global property that controls whether findclMath should search for
#                              64bit or 32bit libs
#-----------------------
# Example Usage:
#
#    find_package(clBLAS REQUIRED)
#    include_directories(${clBLAS_INCLUDE_DIRS})
#
#    add_executable( foo foo.cpp )
#    target_link_libraries(foo ${clBLAS_LIBRARIES})
#
#-----------------------

find_path( clBLAS_INCLUDE_DIR
  NAMES clBLAS.h
  HINTS
  ${CLMATH_BLAS_ROOT}/include
  $ENV{CLMATH_BLAS_ROOT}/include
  DOC "clMath header file path"
)
mark_as_advanced( clBLAS_INCLUDE_DIR )

# Search for 64bit libs if FIND_LIBRARY_USE_LIB64_PATHS is set to true in the global environment, 32bit libs else
get_property( LIB64 GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS )
if( LIB64 )
  message( STATUS "FindclBLAS searching for 64-bit libraries" )
else( )
  message( STATUS "FindclBLAS searching for 32-bit libraries" )
endif( )

find_library( clBLAS_LIBRARY
  NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}clBLAS${CMAKE_STATIC_LIBRARY_SUFFIX}
  HINTS
  ${CLMATH_BLAS_ROOT}/lib
  $ENV{CLMATH_BLAS_ROOT}/lib
  DOC "clMath dynamic library path"
  PATH_SUFFIXES import
)
mark_as_advanced( clBLAS_LIBRARY )

# Set the plural names for clients
set( clBLAS_INCLUDE_DIRS ${clBLAS_INCLUDE_DIR} )
set( clBLAS_LIBRARIES ${clBLAS_LIBRARY} )

include( FindPackageHandleStandardArgs )
FIND_PACKAGE_HANDLE_STANDARD_ARGS( clBLAS DEFAULT_MSG clBLAS_LIBRARIES clBLAS_INCLUDE_DIRS )

if( NOT CLBLAS_FOUND )
  message( STATUS "FindclBLAS failed to properly discover and set up variables" )
  message( STATUS "clBLAS_LIBRARIES: " ${clBLAS_LIBRARIES} )
  message( STATUS "clBLAS_INCLUDE_DIRS: " ${clBLAS_INCLUDE_DIRS} )
endif()
