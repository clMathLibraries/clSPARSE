include( GenerateExportHeader )

# Query the user for which version of OpenCL they wish to build the library for
set( BUILD_CLVERSION "1.2" CACHE STRING "The version of OpenCL we wish to compile the library against" )
set_property( CACHE BUILD_CLVERSION PROPERTY STRINGS 2.0 1.2 1.1 )
message( STATUS "clSPARSE is building using CL interface ='${BUILD_CLVERSION}'" )

# clSPARSE library requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, but we supply our own until 3.1 becomes more prevalent
find_package( OpenCL ${BUILD_CLVERSION} REQUIRED )

# clSPARSE library requires clBLAS
find_package( clBLAS REQUIRED )

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
  ${clBLAS_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
  ${PROJECT_BINARY_DIR}/include
  ${PROJECT_BINARY_DIR}/clsparseTimer
)

set(ocl_kernels_file_name ${PROJECT_BINARY_DIR}/include/kernels/source_provider.cpp)
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
  kernels/control.cl
  kernels/csr2dense.cl
  kernels/csr2coo.cl
  kernels/blas1.cl
  kernels/reduce.cl
  kernels/dot.cl
  kernels/atomic_reduce.cl
  kernels/elementwise_transform.cl
  kernels/matrix_utils.cl
)

set( clSPARSE.source.cpp
  clsparse_init.cpp
  internal/clsparse_control.cpp
  internal/clsparse_validate.cpp
  internal/kernel_cache.cpp
  internal/ocl_type_traits.cpp
  internal/kernel_wrap.cpp
  spmv/clsparse_csrmv.cpp
  transform/clsparse_csr2dense.cpp
  matrixMarket/mm_reader.cpp
  transform/clsparse_coo2csr.cpp
  transform/clsparse_csr2coo.cpp
  blas1/cldense_axpy.cpp
  blas1/cldense_axpby.cpp
  blas1/cldense_scale.cpp
  blas1/cldense_reduce.cpp
  blas1/cldense_dot.cpp
  blas1/reduce_operators.cpp
  blas1/cldense_nrm1.cpp
  blas1/cldense_nrm2.cpp
  blas1/elementwise_operators.cpp
  solvers/solver_control.cpp
  solvers/conjugate_gradients.cpp
)

set( clSPARSE.Headers
  ../include/clSPARSE.h
  ../include/clSPARSE_1x.h
  ../include/clSPARSE_2x.h
  ../include/clsparse.error.hpp
  include/clSPARSE-private.hpp
  include/clSPARSE_1x.hpp
  include/clSPARSE_2x.hpp

  ${PROJECT_BINARY_DIR}/include/clSPARSE.version.h
  internal/clsparse_internal.hpp
  internal/clsparse_control.hpp
  internal/clsparse_validate.hpp
  internal/source_provider.hpp
  internal/kernel_cache.hpp
  internal/ocl_type_traits.hpp
  internal/kernel_wrap.hpp
  internal/data_types/clvector.hpp
  internal/data_types/clarray.hpp
  internal/data_types/clarray_base.hpp
  internal/data_types/reference_base.hpp
  internal/data_types/iterator-base.hpp
  internal/computeRowBlocks.hpp
  spmv/clsparse_csrmv.hpp
  spmv/csrmv_vector/csrmv_vector.hpp
  spmv/csrmv_adaptive/csrmv_adaptive.hpp
  blas1/reduce.hpp
  blas1/atomic_reduce.hpp
  blas1/reduce_operators.hpp
  blas1/commons.hpp
  blas1/cldense_dot.hpp
  blas1/cldense_axpy.hpp
  blas1/cldense_axpby.hpp
  blas1/cldense_nrm1.hpp
  blas1/cldense_nrm2.hpp
  blas1/elementwise_transform.hpp
  blas1/elementwise_operators.hpp
  ${ocl_kernels_file_name}
  solvers/solver_control.hpp
  solvers/preconditioners/preconditioner.hpp
  solvers/preconditioners/diagonal.hpp
  solvers/preconditioners/void.hpp
  solvers/preconditioners/preconditioner_utils.hpp
  solvers/conjugate_gradients.hpp

)

set( clSPARSE.Files ${clSPARSE.source.cpp} ${clSPARSE.Headers} )

# source_group( kernels REGULAR_EXPRESSION ".*cl$" )
source_group( kernels FILES ${clSPARSE.source.cl} )

if( CMAKE_COMPILER_IS_GNUCC )
    add_definitions( -pedantic )
endif( )

add_compiler_export_flags( )
add_library( clSPARSE SHARED ${clSPARSE.Files} ${clSPARSE.source.cl})

# PRIVATE linking prevents transitive library linking of the clBLAS libraries
target_link_libraries( clSPARSE PRIVATE ${OPENCL_LIBRARIES} ${clBLAS_LIBRARIES} )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( clSPARSE )

set_target_properties( clSPARSE PROPERTIES VERSION ${clSPARSE_VERSION} SOVERSION ${clSPARSE_SOVERSION} )
set_target_properties( clSPARSE PROPERTIES DEBUG_POSTFIX d )
set_target_properties( clSPARSE PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# The following installs the basic files of the library
install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clsparse_export.h
  DESTINATION
    include
)

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
set( ConfigPackageLocation ${LIB_INSTALL_DIR}/clSPARSE/cmake )

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

install( EXPORT clSPARSE-Targets
  DESTINATION
    ${ConfigPackageLocation}
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/clSPARSEConfigVersion.cmake
  DESTINATION
    ${ConfigPackageLocation} )
