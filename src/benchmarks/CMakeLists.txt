
set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
set( Boost_NO_SYSTEM_PATHS ON )
set( Boost_ADDITIONAL_VERSIONS 1.57.0 1.57 )
# set( Boost_DEBUG ON )

find_package( Boost COMPONENTS program_options filesystem regex system )

if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  add_definitions( -std=c++11 )
endif( )

include_directories(
  ${Boost_INCLUDE_DIRS}
)

add_subdirectory( clsparse-bench )

# Only enable/build the cuda benchmark if CUDA is detected on the platform
find_package( CUDA )

if( CUDA_FOUND )
  add_subdirectory( cusparse-bench )
endif( )
