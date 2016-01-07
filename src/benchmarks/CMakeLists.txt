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
endif( )

# set( Boost_DEBUG ON )
set( Boost_USE_MULTITHREADED ON )
set( Boost_DETAILED_FAILURE_MSG ON )
set( Boost_ADDITIONAL_VERSIONS 1.60.0 1.60 1.59.0 1.59 1.58.0 1.58 1.57.0 1.57 )

# On windows, Boost prefers to link statically unless the user defines BOOST_ALL_DYN_LINK
# Therefore, we prefer linking  static libs first to avoid extra #defines
set( Boost_USE_STATIC_LIBS ON )
find_package( Boost COMPONENTS program_options filesystem regex system )

# If static libraries are not found, search again for shared libraries
if( NOT Boost_FOUND )
  set( Boost_USE_STATIC_LIBS OFF )
  find_package( Boost COMPONENTS program_options filesystem regex system )
endif( )

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
