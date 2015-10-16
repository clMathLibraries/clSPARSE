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

find_package( CUDA REQUIRED )

set( CLIENT_SRC
  src/main.cpp
  src/statisticalTimer.cpp
  src/mm_reader.cpp
  )

set( CLIENT_HEADER
  include/statisticalTimer.h
  include/mm_reader.hpp
  include/io-exception.hpp
  functions/cufunc_common.hpp
  functions/cufunc_xSpMdV.hpp
  functions/cufunc_xCsr2dense.hpp
  functions/cufunc_xCsr2Coo.hpp
  functions/cufunc_xDense2Csr.hpp
  functions/cufunc_xCoo2Csr.hpp
  functions/cufunc_xSpMSpM.hpp
)

include_directories(
  ${CUDA_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/library
  ${PROJECT_BINARY_DIR}/library
  ${CMAKE_CURRENT_SOURCE_DIR}
)

CUDA_ADD_EXECUTABLE( cusparse-bench ${CLIENT_SRC} ${CLIENT_HEADER} )
CUDA_ADD_CUBLAS_TO_TARGET( cusparse-bench )
target_link_libraries( cusparse-bench ${Boost_LIBRARIES} ${CUDA_cusparse_LIBRARY} )
set_target_properties( cusparse-bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( cusparse-bench PROPERTIES VERSION ${clSPARSE_VERSION} )
set_property( TARGET cusparse-bench PROPERTY FOLDER "Benchmarks")

# install( TARGETS cusparse.bench
#   RUNTIME DESTINATION bin${SUFFIX_BIN}
#   LIBRARY DESTINATION lib${SUFFIX_LIB}
#   ARCHIVE DESTINATION lib${SUFFIX_LIB}
# )
