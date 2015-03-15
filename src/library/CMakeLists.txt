include( GenerateExportHeader )

# clSPARSE library requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

# clSPARSE library requires clBLAS
find_package( clBLAS REQUIRED )

set( clSPARSE_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${PROJECT_BINARY_DIR}/include
  ${clSPARSE_INCLUDE_DIRS}
  ${clBLAS_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
)

set( ocl_kernels_file_name ${PROJECT_BINARY_DIR}/include/kernels/program_sources.h )

add_custom_command(
  OUTPUT "${ocl_kernels_file_name}"
  COMMAND ${CMAKE_COMMAND}
    -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
    -DOUTPUT="${ocl_kernels_file_name}"
    -P "${PROJECT_SOURCE_DIR}/cmake/cl2c.cmake"
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
  clsparse_init.c
  internal/hdl_list.c
  internal/clsparse_internal.c
  internal/clsparse_control.c
  internal/clsparse_sources.c
  internal/clsparse_validate.c
  simple/clsparse_simple.c
  spmv/clsparse_csrmv.c
)

set( clSPARSE.Headers
  ../include/clSPARSE.h
  ${PROJECT_BINARY_DIR}/include/clSPARSE.version.h
  internal/hdl_list.h
  internal/clsparse_internal.h
  internal/clsparse_control.h
  internal/clsparse_sources.h
  internal/clsparse_validate.h
  ${ocl_kernels_file_name}
)

set( clSPARSE.Files ${clSPARSE.source.cpp} ${clSPARSE.Headers} ${clSPARSE.source.cl} )

if( CMAKE_COMPILER_IS_GNUCC )
    set( CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -std=c99 )
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
