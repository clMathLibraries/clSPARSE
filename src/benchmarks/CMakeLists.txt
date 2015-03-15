set( Boost_USE_MULTITHREADED ON )
set( Boost_USE_STATIC_LIBS   ON )
set( Boost_DETAILED_FAILURE_MSG   ON )
# set( Boost_DEBUG ON )

find_package( Boost COMPONENTS program_options filesystem system regex )

# Benchmarks requires OpenCL
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more prevalent
find_package( OpenCL REQUIRED )

if( CMAKE_COMPILER_IS_GNUCC )
  set( CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} )
endif( )

include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/tests/resources
  ${PROJECT_BINARY_DIR}/library
  common
  ${Boost_INCLUDE_DIRS}
  ${OPENCL_INCLUDE_DIRS}
)

set(headers
  common/Timer.h
  common/Find.h
  common/MatrixUtils.h
  common/Executor.h
  common/Params.h
  common/MatrixStatistics.h
)

set(sources
  common/Timer.cpp
)

set(BENCHMARKS
  csrmv_benchmark
)

foreach( benchmark ${BENCHMARKS} )
  add_executable( ${benchmark} ${benchmark}.cpp ${sources} ${headers} )
  target_link_libraries( ${benchmark} clSPARSE ${Boost_LIBRARIES} ${OPENCL_LIBRARIES} )
  set_property( TARGET ${benchmark} PROPERTY FOLDER "Benchmarks")
endforeach( )
