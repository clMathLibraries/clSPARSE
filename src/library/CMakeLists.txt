include( GenerateExportHeader )

# clSPARSE library requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

# clSPARSE library requires clBLAS
find_package( clBLAS REQUIRED )

if( CMAKE_COMPILER_IS_GNUCXX )
  add_definitions(
      -std=c++11 # support for new cpp standard
  )
endif( )

add_definitions(
    -D__CL_ENABLE_EXCEPTIONS
)

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
    -P "${PROJECT_SOURCE_DIR}/cmake/cl2cpp.cmake"
)

# List the names of common files to compile across all platforms
set( clSPARSE.source.cl
  kernels/csrmv_alpha1.cl
  kernels/csrmv_alpha1_beta0.cl
  kernels/csrmv_beta0.cl
  kernels/csrmv_beta1.cl
  kernels/csrmv_general.cl
  kernels/scale.cl
)

set( clSPARSE.source.cpp
  clsparse_init.cpp
  internal/clsparse_control.cpp
  internal/clsparse_validate.cpp
  internal/kernel_cache.cpp
  internal/ocl_type_traits.cpp
  internal/kernel_wrap.cpp
  simple/clsparse_simple.cpp
  spmv/clsparse_csrmv.cpp

)

set( clSPARSE.Headers
  ../include/clSPARSE.h
  ${PROJECT_BINARY_DIR}/include/clSPARSE.version.h
   internal/clsparse_internal.hpp
  internal/clsparse_control.hpp
  internal/clsparse_validate.hpp
  internal/source_provider.hpp
  internal/kernel_cache.hpp
  internal/ocl_type_traits.hpp
  internal/kernel_wrap.hpp
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
  ../include/clSPARSE.h
  ${PROJECT_BINARY_DIR}/include/clSPARSE.version.h
  ${CMAKE_CURRENT_BINARY_DIR}/clsparse_export.h
  DESTINATION
    include
)
