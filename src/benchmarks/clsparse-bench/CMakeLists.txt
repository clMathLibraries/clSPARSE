# Benchmarks requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

set( clsparse.bench.src
  src/main.cpp
  src/statisticalTimer.cpp
)

set( clsparse.bench.hdr
  include/statisticalTimer.h
  functions/clfunc_common.hpp
  functions/clfunc_xSpMdV.hpp
  functions/clfunc_xCGM.hpp
  )

# Having problems on build server, compiling gtest headers with -pedantic; disabling detection of long long
# http://code.google.com/p/googletest/issues/detail?id=334
# if( CMAKE_COMPILER_IS_GNUCXX )
#   add_definitions( -Wno-long-long )
# endif( )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_BINARY_DIR}/library
  ${PROJECT_BINARY_DIR}/clsparseTimer
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${OPENCL_INCLUDE_DIRS}
 )

add_executable( clsparse.bench ${clsparse.bench.src} ${clsparse.bench.hdr} )
target_link_libraries( clsparse.bench ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} ${CMAKE_DL_LIBS} clSPARSE )
set_target_properties( clsparse.bench PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
set_target_properties( clsparse.bench PROPERTIES VERSION ${clSPARSE_VERSION} )
set_property( TARGET clsparse.bench PROPERTY FOLDER "Benchmarks")

# CPack configuration; include the executable into the package
install( TARGETS clsparse.bench
  RUNTIME DESTINATION bin${SUFFIX_BIN}
  LIBRARY DESTINATION lib${SUFFIX_LIB}
  ARCHIVE DESTINATION lib${SUFFIX_LIB}
)
