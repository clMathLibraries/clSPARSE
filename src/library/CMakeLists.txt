# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
# Copyright 2015 Vratis, Ltd.
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

include( GenerateExportHeader )

# Query the user for which version of OpenCL they wish to build the library for
set( BUILD_CLVERSION "1.2" CACHE STRING "The version of OpenCL we wish to compile the library against" )
set_property( CACHE BUILD_CLVERSION PROPERTY STRINGS 2.0 1.2 1.1 )
message( STATUS "clSPARSE is building using CL interface ='${BUILD_CLVERSION}'" )

# clSPARSE library requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, but we supply our own until 3.1 becomes more prevalent
find_package( OpenCL ${BUILD_CLVERSION} REQUIRED )

# clSPARSE library requires clBLAS
# find_package( clBLAS REQUIRED )

if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  add_definitions( -std=c++11 )
endif( )

if( BUILD_CLVERSION VERSION_EQUAL "2.0" )
    add_definitions( -DBUILD_CLVERSION=200 )
elseif( BUILD_CLVERSION VERSION_EQUAL "1.2" )
    add_definitions( -DBUILD_CLVERSION=120 )
elseif( BUILD_CLVERSION VERSION_EQUAL "1.1" )
    add_definitions( -DBUILD_CLVERSION=110 )
endif( )

#add_definitions( -D__CL_ENABLE_EXCEPTIONS )

set( clSPARSE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${clSPARSE_INCLUDE_DIRS}
# ${clBLAS_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/clsparseTimer
)

set(ocl_kernels_file_name ${PROJECT_BINARY_DIR}/include/kernels/source-provider.cpp)
add_custom_command(
  OUTPUT "${ocl_kernels_file_name}"
  COMMAND ${CMAKE_COMMAND}
    -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    -DOUTPUT="${ocl_kernels_file_name}"
    -P "${PROJECT_SOURCE_DIR}/cmake/cl2cpp_raw.cmake"
)

# List the names of common files to compile across all platforms
set( clSPARSE.source.cl
  kernels/csrmv_general.cl
  kernels/csrmv_adaptive.cl
  kernels/csrmm_general.cl
  kernels/control.cl
  kernels/csr2dense.cl
  kernels/csr2coo.cl
  kernels/blas1.cl
  kernels/reduce.cl
  kernels/dot.cl
  kernels/atomic_reduce.cl
  kernels/elementwise_transform.cl
  kernels/matrix_utils.cl
  kernels/scan_kernels.cl
  kernels/reduce_by_key.cl
  kernels/sort_by_key_common.cl
  kernels/sort_by_key_int.cl
  kernels/sort_by_key_uint.cl
  kernels/prescan_scatter.cl
  kernels/dense2csr.cl
)

set( clSPARSE.source.cpp
  clsparse-init.cpp
  io/mm-reader.cpp
)

set( clSPARSE.source.internal
  internal/clsparse-control.cpp
  internal/clsparse-validate.cpp
  internal/kernel-cache.cpp
  internal/ocl-type-traits.cpp
  internal/kernel-wrap.cpp
)

set( clSPARSE.source.transform
  transform/clsparse-csr2dense.cpp
  transform/clsparse-coo2csr.cpp
  transform/clsparse-csr2coo.cpp
  transform/scan.cpp
  transform/reduce-by-key.cpp
  transform/sort-by-key.cpp
  transform/clsparse-coo2csr-GPU.cpp
  transform/clsparse-dense2csr.cpp
)

set( clSPARSE.source.blas1
  blas1/cldense-axpy.cpp
  blas1/cldense-axpby.cpp
  blas1/cldense-scale.cpp
  blas1/cldense-reduce.cpp
  blas1/cldense-dot.cpp
  blas1/reduce-operators.cpp
  blas1/cldense-nrm1.cpp
  blas1/cldense-nrm2.cpp
  blas1/elementwise-operators.cpp
)

set( clSPARSE.source.blas2
  blas2/clsparse-csrmv.cpp
)

set( clSPARSE.source.blas3
  blas3/clsparse-csrmm.cpp
)

set( clSPARSE.source.solvers
  solvers/solver-control.cpp
  solvers/conjugate-gradients.cpp
  solvers/biconjugate-gradients-stabilized.cpp
)

set( clSPARSE.Headers.public
  ../include/clSPARSE.h
  ../include/clSPARSE-1x.h
  ../include/clSPARSE-2x.h
  ../include/clSPARSE-error.h
  ${PROJECT_BINARY_DIR}/include/clSPARSE-version.h
)

set( clSPARSE.Headers.private
  include/clSPARSE-1x.hpp
  include/clSPARSE-2x.hpp
  include/clSPARSE-private.hpp
)

set( clSPARSE.Headers.blas1
  blas1/reduce.hpp
  blas1/atomic-reduce.hpp
  blas1/reduce-operators.hpp
  blas1/commons.hpp
  blas1/cldense-dot.hpp
  blas1/cldense-axpy.hpp
  blas1/cldense-axpby.hpp
  blas1/cldense-nrm1.hpp
  blas1/cldense-nrm2.hpp
  blas1/elementwise-transform.hpp
  blas1/elementwise-operators.hpp
)

set( clSPARSE.Headers.blas2
  blas2/clsparse-csrmv.hpp
  blas2/csrmv-vector.hpp
  blas2/csrmv-adaptive.hpp
)

set( clSPARSE.Headers.blas3
  blas3/clsparse-csrmm.hpp
)

set( clSPARSE.Headers.solvers
  solvers/solver-control.hpp
  solvers/preconditioners/preconditioner.hpp
  solvers/preconditioners/diagonal.hpp
  solvers/preconditioners/void.hpp
  solvers/preconditioners/preconditioner_utils.hpp
  solvers/conjugate-gradients.hpp
  solvers/biconjugate-gradients-stabilized.hpp
)

set( clSPARSE.Headers.internal
  internal/clsparse-internal.hpp
  internal/clsparse-control.hpp
  internal/clsparse-validate.hpp
  internal/source-provider.hpp
  internal/kernel-cache.hpp
  internal/ocl-type-traits.hpp
  internal/kernel-wrap.hpp
  internal/data-types/clvector.hpp
  internal/data-types/clarray.hpp
  internal/data-types/clarray-base.hpp
  internal/data-types/reference-base.hpp
  internal/data-types/iterator-base.hpp
  internal/computeRowBlocks.hpp
  transform/transform-kernels.h
  ${ocl_kernels_file_name}
)

# source_group( kernels REGULAR_EXPRESSION ".*cl$" )
# Group header files into nice bundles
source_group( "Header Files\\Kernels" FILES ${clSPARSE.source.cl} )
source_group( "Header Files\\Public" FILES ${clSPARSE.Headers.public} )
source_group( "Header Files\\Private" FILES ${clSPARSE.Headers.private} )
source_group( "Header Files\\Blas1" FILES ${clSPARSE.Headers.blas1} )
source_group( "Header Files\\Blas2" FILES ${clSPARSE.Headers.blas2} )
source_group( "Header Files\\Blas3" FILES ${clSPARSE.Headers.blas3} )
source_group( "Header Files\\Solvers" FILES ${clSPARSE.Headers.solvers} )
source_group( "Header Files\\Internal" FILES ${clSPARSE.Headers.internal} )

# Group source files into nice bundles
source_group( "Source Files\\Internal" FILES ${clSPARSE.source.internal} )
source_group( "Source Files\\Transform" FILES ${clSPARSE.source.transform} )
source_group( "Source Files\\Blas1" FILES ${clSPARSE.source.blas1} )
source_group( "Source Files\\Blas2" FILES ${clSPARSE.source.blas2} )
source_group( "Source Files\\Blas3" FILES ${clSPARSE.source.blas3} )
source_group( "Source Files\\Solvers" FILES ${clSPARSE.source.solvers} )

if( CMAKE_COMPILER_IS_GNUCXX )
    add_definitions( -pedantic )
endif( )

# Query the user for which version of OpenCL they wish to build the library for
set( clSPARSE_LIBRARY_TYPE "SHARED" CACHE STRING "Build the clSPARSE library as SHARED or STATIC build types" )
set_property( CACHE clSPARSE_LIBRARY_TYPE PROPERTY STRINGS SHARED STATIC )
message( STATUS "clSPARSE will build as a '${clSPARSE_LIBRARY_TYPE}' library" )

add_compiler_export_flags( )
add_library( clSPARSE ${clSPARSE_LIBRARY_TYPE}
  ${clSPARSE.source.cpp}
  ${clSPARSE.source.internal}
  ${clSPARSE.source.transform}
  ${clSPARSE.source.blas1}
  ${clSPARSE.source.blas2}
  ${clSPARSE.source.blas3}
  ${clSPARSE.source.solvers}
  ${clSPARSE.Headers.public}
  ${clSPARSE.Headers.private}
  ${clSPARSE.Headers.blas1}
  ${clSPARSE.Headers.blas2}
  ${clSPARSE.Headers.blas3}
  ${clSPARSE.Headers.solvers}
  ${clSPARSE.Headers.internal}
  ${clSPARSE.source.cl}
  )

# PRIVATE linking prevents transitive library linking of the clBLAS libraries
target_link_libraries( clSPARSE PRIVATE ${OPENCL_LIBRARIES} ) #${clBLAS_LIBRARIES} )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( clSPARSE )

set_target_properties( clSPARSE PROPERTIES VERSION ${clSPARSE_VERSION} SOVERSION ${clSPARSE_SOVERSION} )
set_target_properties( clSPARSE PROPERTIES DEBUG_POSTFIX d )
set_target_properties( clSPARSE PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# Following Boost conventions of prefixing 'lib' on static built libraries, across all platforms
if( ${clSPARSE_LIBRARY_TYPE} STREQUAL "STATIC" )
  set_target_properties( clSPARSE PROPERTIES PREFIX "lib" ) # global CMAKE_STATIC_LIBRARY_PREFIX
endif( )

# I think this would all be a bit easier if clBLAS started exporting its targets
# similar to what clsparse is doing
#if( MSVC_IDE )
#  set( clBLAS_LIBRARY_SHARED ${CLMATH_BLAS_ROOT}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}clBLAS${CMAKE_SHARED_LIBRARY_SUFFIX} )
#  set( clblasCommand COMMAND ${CMAKE_COMMAND} -E copy_if_different ${clBLAS_LIBRARY_SHARED} ${PROJECT_BINARY_DIR}/staging/${CMAKE_CFG_INTDIR} )
#else( )
#  # UNIX based systems are using a SOLINK, link to it instead
#  set( clBLAS_LIBRARY_SOLINK ${CMAKE_SHARED_LIBRARY_PREFIX}clBLAS.2${CMAKE_SHARED_LIBRARY_SUFFIX} )
#  set( clblasCommand COMMAND ${CMAKE_COMMAND} -E create_symlink ${clBLAS_LIBRARY} ${PROJECT_BINARY_DIR}/staging/${CMAKE_CFG_INTDIR}/${clBLAS_LIBRARY_SOLINK} )
#endif( )

#add_custom_command( TARGET clSPARSE
#                   POST_BUILD
#                   ${clblasCommand}
##                   [BYPRODUCTS [files...]]
##                   [WORKING_DIRECTORY dir]
#                   COMMENT "Copying clBLAS library dependency into staging directory"
#                   VERBATIM
#                   )

# CPack configuration; include the executable into the package
install( TARGETS clSPARSE
  EXPORT clSPARSE-Targets
  RUNTIME DESTINATION bin${SUFFIX_BIN}
  LIBRARY DESTINATION lib${SUFFIX_LIB}
  ARCHIVE DESTINATION lib${SUFFIX_LIB}
  INCLUDES DESTINATION include
)

# The following is cmake code to generate a config file package for clSPARSE
include( CMakePackageConfigHelpers )

set( LIB_INSTALL_DIR lib${SUFFIX_LIB} )
set( INCLUDE_INSTALL_DIR include )
set( ConfigPackageLocation ${LIB_INSTALL_DIR}/cmake/clSPARSE )

configure_package_config_file(
  clSPARSEConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfig.cmake
  INSTALL_DESTINATION ${ConfigPackageLocation}
  PATH_VARS INCLUDE_INSTALL_DIR LIB_INSTALL_DIR
)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfigVersion.cmake
  VERSION ${clSPARSE_VERSION}
  COMPATIBILITY SameMajorVersion
)

# The following installs the basic files of the library
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clsparse_export.h
  DESTINATION
    include
)

# This generates the files that defines the import targets
install( EXPORT clSPARSE-Targets
  DESTINATION
    ${ConfigPackageLocation}
)

#Copy the config files generated by configure_package_config_file & write_basic_package_version_file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfigVersion.cmake
  DESTINATION
    ${ConfigPackageLocation} )
