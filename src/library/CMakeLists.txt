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

add_definitions( -D__CL_ENABLE_EXCEPTIONS )

set( clSPARSE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/include
  ${clSPARSE_INCLUDE_DIRS}
  ${clBLAS_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
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
  kernels/csrmv_alpha1.cl
  kernels/csrmv_alpha1_beta0.cl
  kernels/csrmv_beta0.cl
  kernels/csrmv_beta1.cl
  kernels/csrmv_general.cl
  kernels/csrmv_adaptive.cl
  kernels/control.cl
  kernels/csr2dense.cl
  kernels/csr2coo.cl
  kernels/blas1.cl
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
  blas1/clsparse_axpy.cpp
  blas1/clsparse_scale.cpp
)

set( clSPARSE.Headers
  ../include/clSPARSE.h
  ../include/clSPARSE_1x.h
  ../include/clSPARSE_2x.h
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
  spmv/csrmv_vector/csrmv_vector.hpp
  spmv/csrmv_vector/csrmv_vector_impl.hpp
  spmv/csrmv_adaptive/csrmv_adaptive.hpp
  ${ocl_kernels_file_name}
)

set( clSPARSE.Files ${clSPARSE.source.cpp} ${clSPARSE.Headers} ${clSPARSE.source.cl} )

if( CMAKE_COMPILER_IS_GNUCC )
    add_definitions( -pedantic )
endif( )

add_compiler_export_flags( )
add_library( clSPARSE SHARED ${clSPARSE.Files} )

# PRIVATE linking prevents transitive library linking of the clBLAS libraries
target_link_libraries( clSPARSE PRIVATE ${OPENCL_LIBRARIES} ${clBLAS_LIBRARIES} )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( clSPARSE )

set_target_properties( clSPARSE PROPERTIES VERSION ${clSPARSE_VERSION} )
set_target_properties( clSPARSE PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )

# CPack configuration; include the executable into the package
install( TARGETS clSPARSE
  RUNTIME DESTINATION bin${SUFFIX_BIN}
  LIBRARY DESTINATION lib${SUFFIX_LIB}
  ARCHIVE DESTINATION lib${SUFFIX_LIB}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clsparse_export.h
  DESTINATION
    include
)
