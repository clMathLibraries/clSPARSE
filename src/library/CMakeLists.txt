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
  kernels/csrmm.cl
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
  matrixMarket/mm_reader.cpp
)

set( clSPARSE.source.internal
  internal/clsparse_control.cpp
  internal/clsparse_validate.cpp
  internal/kernel_cache.cpp
  internal/ocl_type_traits.cpp
  internal/kernel_wrap.cpp
)

set( clSPARSE.source.transform
  transform/clsparse_csr2dense.cpp
  transform/clsparse_coo2csr.cpp
  transform/clsparse_csr2coo.cpp
)

set( clSPARSE.source.blas1
  blas1/cldense_axpy.cpp
  blas1/cldense_axpby.cpp
  blas1/cldense_scale.cpp
  blas1/cldense_reduce.cpp
  blas1/cldense_dot.cpp
  blas1/reduce_operators.cpp
  blas1/cldense_nrm1.cpp
  blas1/cldense_nrm2.cpp
  blas1/elementwise_operators.cpp
)

set( clSPARSE.source.blas2
  spmv/clsparse_csrmv.cpp
)

set( clSPARSE.source.blas3
  blas3/clsparse-csrmm.cpp
)

set( clSPARSE.source.solvers
  solvers/solver_control.cpp
  solvers/conjugate_gradients.cpp
  solvers/biconjugate-gradients-stabilized.cpp
)

set( clSPARSE.Headers.public
  ../include/clSPARSE.h
  ../include/clSPARSE_1x.h
  ../include/clSPARSE_2x.h
  ../include/clsparse.error.hpp
  ${PROJECT_BINARY_DIR}/include/clSPARSE.version.h
)

set( clSPARSE.Headers.private
  include/clSPARSE_1x.hpp
  include/clSPARSE_2x.hpp
  include/clSPARSE-private.hpp
)

set( clSPARSE.Headers.blas1
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
)

set( clSPARSE.Headers.blas2
  spmv/clsparse_csrmv.hpp
  spmv/csrmv_vector/csrmv_vector.hpp
  spmv/csrmv_adaptive/csrmv_adaptive.hpp
)

set( clSPARSE.Headers.blas3
  blas3/clsparse-csrmm.hpp
)

set( clSPARSE.Headers.solvers
  solvers/solver_control.hpp
  solvers/preconditioners/preconditioner.hpp
  solvers/preconditioners/diagonal.hpp
  solvers/preconditioners/void.hpp
  solvers/preconditioners/preconditioner_utils.hpp
  solvers/conjugate_gradients.hpp
  solvers/biconjugate-gradients-stabilized.hpp
)

set( clSPARSE.Headers.internal
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
target_link_libraries( clSPARSE PRIVATE ${OPENCL_LIBRARIES} ${clBLAS_LIBRARIES} )

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
if( MSVC_IDE )
  set( clBLAS_LIBRARY_SHARED ${CLMATH_BLAS_ROOT}/bin/${CMAKE_SHARED_LIBRARY_PREFIX}clBLAS${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set( clblasCommand COMMAND ${CMAKE_COMMAND} -E copy_if_different ${clBLAS_LIBRARY_SHARED} ${PROJECT_BINARY_DIR}/staging/${CMAKE_CFG_INTDIR} )
else( )
  # UNIX based systems are using a SOLINK, link to it instead
  set( clBLAS_LIBRARY_SOLINK ${CMAKE_SHARED_LIBRARY_PREFIX}clBLAS.2${CMAKE_SHARED_LIBRARY_SUFFIX} )
  set( clblasCommand COMMAND ${CMAKE_COMMAND} -E create_symlink ${clBLAS_LIBRARY} ${PROJECT_BINARY_DIR}/staging/${CMAKE_CFG_INTDIR}/${clBLAS_LIBRARY_SOLINK} )
endif( )

add_custom_command( TARGET clSPARSE
                   POST_BUILD
                   ${clblasCommand}
#                   [BYPRODUCTS [files...]]
#                   [WORKING_DIRECTORY dir]
                   COMMENT "Copying clBLAS library dependency into staging directory"
                   VERBATIM
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
