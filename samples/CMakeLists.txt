cmake_minimum_required( VERSION 2.8.12 )
project(samples)

message( STATUS "BUILDING EXAMPLES")
message( STATUS "BUILD64: ${BUILD64}")

set( CLSPARSE_LIB_SUFFIX "")
if( ${BUILD64} )
   set( CLSPARSE_LIB_SUFFIX "64")
   set_property( GLOBAL PROPERTY FIND_LIBRARY_USE_LIB64_PATHS TRUE )
endif( )

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find clSPARSE library
find_package( clSPARSE REQUIRED CONFIG )
message( STATUS "clSPARSE INCLUDES = ${clSPARSE_INCLUDE_DIR}")

# Find OpenCL
find_package( OpenCL ${BUILD_CLVERSION} REQUIRED )
message( STATUS "OPENCL_INCLUDE_DIRS = ${OPENCL_INCLUDE_DIRS}")

include_directories( ${OPENCL_INCLUDE_DIRS} )

# No we are ready to generate the samples

set (SAMPLES_CPP
sample-initialization
sample-axpy
sample-spmv
sample-cg
)

set (SAMPLES_C
sample-initialization-c
sample-norm1-c
)

set(EXAMPLE_TARGETS)

foreach( sample ${SAMPLES_CPP} )
  add_executable( ${sample} ${sample}.cpp)
  target_link_libraries( ${sample} clSPARSE ${CMAKE_DL_LIBS} ${OPENCL_LIBRARIES} )
  if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
   target_compile_options( ${sample} PUBLIC -std=c++11 )
  endif( )
  set_target_properties( ${sample} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${sample} PROPERTY FOLDER "Samples" )
  # collect targets for installation
  list(APPEND EXAMPLE_TARGETS ${sample})
endforeach( )

foreach( sample ${SAMPLES_C} )
  add_executable( ${sample} ${sample}.c)
  target_link_libraries( ${sample} clSPARSE ${OPENCL_LIBRARIES})
  set_target_properties( ${sample} PROPERTIES VERSION ${clSPARSE_VERSION} )
  set_target_properties( ${sample} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
  set_property( TARGET ${sample} PROPERTY FOLDER "Samples" )
# collect targets for installation
  list(APPEND EXAMPLE_TARGETS ${sample})
endforeach( )

#Installation step
message (STATUS "DEST ${CMAKE_PREFIX_PATH}/Samples")
message (STATUS "TARGETS ${EXAMPLE_TARGETS}")
install(TARGETS ${EXAMPLE_TARGETS} DESTINATION  ${CMAKE_PREFIX_PATH}/Samples)
