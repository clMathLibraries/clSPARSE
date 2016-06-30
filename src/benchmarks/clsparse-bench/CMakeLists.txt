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

# Benchmarks requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

set( clsparse.bench.src
  src/main.cpp
  src/statisticalTimer.cpp
)

set( clsparse.bench.hdr
  include/statisticalTimer.h
  include/io-exception.hpp
  functions/clfunc_common.hpp
  functions/clfunc_xSpMdV.hpp
  functions/clfunc-xSpMdM.hpp
  functions/clfunc_xCG.hpp
  functions/clfunc_xBiCGStab.hpp
  functions/clfunc_xCoo2Csr.hpp
  functions/clfunc_xCsr2Coo.hpp
  functions/clfunc_xDense2Csr.hpp
  functions/clfunc_xCsr2Dense.hpp
  functions/clfunc_xSpMSpM.hpp
  )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/library
  ${PROJECT_BINARY_DIR}/clsparseTimer
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPENCL_INCLUDE_DIRS}
 )

add_executable( clsparse-bench ${clsparse.bench.src} ${clsparse.bench.hdr} )
target_link_libraries( clsparse-bench ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} ${CMAKE_DL_LIBS} clSPARSE )
set_target_properties( clsparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( clsparse-bench PROPERTIES VERSION ${clSPARSE_VERSION} )
set_property( TARGET clsparse-bench PROPERTY FOLDER "Benchmarks")

if(NOT USE_SYSTEM_CL2HPP)
    add_dependencies(clsparse-bench cl2hpp)
endif(NOT USE_SYSTEM_CL2HPP)

# CPack configuration; include the executable into the package
# install( TARGETS clsparse.bench
#   RUNTIME DESTINATION bin${SUFFIX_BIN}
#   LIBRARY DESTINATION lib${SUFFIX_LIB}
#   ARCHIVE DESTINATION lib${SUFFIX_LIB}
# )
