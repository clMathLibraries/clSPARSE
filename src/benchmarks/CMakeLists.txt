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


message( STATUS "Configuring benchmarks sub-dir" )

# If user provides a specific boost path through command line, make sure to
# use that
if( DEFINED BOOST_ROOT OR DEFINED ENV{BOOST_ROOT} )
  set( Boost_NO_SYSTEM_PATHS ON )
  set( Boost_NO_BOOST_CMAKE ON )

  # Workaround for a somewhat limiting find_package( Boost ) on MSVC platforms.
  # Boost uses the prefix 'lib' for static libraries on windows, but it will
  # not search with that prefix unless Boost_USE_STATIC_LIBS is set.  However, other
  # than a priori knowledge, a user does not know to set Boost_USE_STATIC_LIBS.
  # Attempt to do a basic check on windows, which may/may not work in all cases
  if( MSVC )
    file( GLOB Boost.static.env.libs RELATIVE "$ENV{BOOST_ROOT}/lib" "$ENV{BOOST_ROOT}/lib/lib*.lib" )
    file( GLOB Boost.static.define.libs RELATIVE "${BOOST_ROOT}/lib" "${BOOST_ROOT}/lib/lib*.lib" )
    list( LENGTH Boost.static.env.libs num.static.env.libs )
    list( LENGTH Boost.static.define.libs num.static.define.libs )
    math( EXPR Boost.num.static.libs "${num.static.env.libs} + ${num.static.define.libs} " )
    if( Boost.num.static.libs )
      message( STATUS "Detected MSVC boost static libs" )
      set( Boost_USE_STATIC_LIBS ON )
    endif( )
  endif( )
endif( )

set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
set( Boost_ADDITIONAL_VERSIONS 1.58.0 1.58 1.57.0 1.57 )
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
