include( GenerateExportHeader )

# clSPARSE library has a hard dependency on clBLAS
find_package( clBLAS REQUIRED )

set( CLSPARSE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include )

include_directories(
  ${CLSPARSE_INCLUDE_DIRS}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${OPENCL_INCLUDE_DIRS}
  ${CLBLAS_INCLUDE_DIRS}
)

set( ocl_kernels_file_name src/kernels/program_sources.h )

add_custom_command(
   OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file_name}"
   COMMAND ${CMAKE_COMMAND}
                -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
                -DOUTPUT="${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file_name}"
                -P "${CMAKE_SOURCE_DIR}/cmake/cl2c.cmake"
#   DEPENDS ${kernels} "${CMAKE_SOURCE_DIR}/cmake/cl2c.cmake"
)

# List the names of common files to compile across all platforms
set( clSPARSE.source.cl
  csrmv_alpha1.cl
  csrmv_alpha1.beta0.cl
  csrmv_beta0.cl
  csrmv_beta1.cl
  csrmv_general.cl
  scale.cl
)

set( clSPARSE.source.cpp
  src/clsparse_init.c
  src/internal/hdl_list.c
  src/internal/clsparse_internal.c
  src/internal/clsparse_control.c
  src/simple/clsparse_simple.c
  src/spmv/clsparse_csrmv.c
)

set( clSPARSE.Headers
  include/clSPARSE.h
  src/clSPARSE.version.h
  src/internal/hdl_list.h
  src/internal/clsparse_internal.h
  src/internal/clsparse_control.h
  src/internal/clsparse_sources.h
  src/internal/clsparse_validate.h
  ${CMAKE_CURRENT_BINARY_DIR}/${ocl_kernels_file_name}
)

set( clSPARSE.Files ${clSPARSE.source.cpp} ${clSPARSE.Headers} ${clSPARSE.source.cl} )

if( CMAKE_COMPILER_IS_GNUCC )
    set( CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -std=c99 )
    add_definitions( -pedantic )
endif( )

add_compiler_export_flags( )
add_library( clSPARSE SHARED ${clSPARSE.Files} )

# PRIVATE linking prevents transitive library linking of the clBLAS libraries
target_link_libraries( clSPARSE PRIVATE ${OPENCL_LIBRARIES} ${CLBLAS_LIBRARIES} )

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
