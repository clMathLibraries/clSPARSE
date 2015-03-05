project(benchmarks)

cmake_minimum_required(VERSION 2.8)

find_package(Boost COMPONENTS
    program_options
    system
    regex
    filesystem
)

find_package(clBLAS)

message(STATUS "clBLAS include ${clBLAS_INCLUDE_DIR}")
message(STATUS "clBLAS lib ${clBLAS_LIB}")

if( NOT WIN32 )
    set(CMAKE_CXX_FLAGS  ${CMAKE_CXX_FLAGS} -std=c++11)
endif( )

message(STATUS "Boost libraries:" ${Boost_LIBRARIES})
message(STATUS "OpenCL headers:" ${OPENCL_INCLUDE_DIR})

include_directories(${Boost_INCLUDE_DIR})
include_directories(../library/include)
include_directories(../tests/resources)
include_directories(common)
include_directories(${clBLAS_INCLUDE_DIR})

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

foreach(benchmark ${BENCHMARKS})
        add_executable(${benchmark} ${benchmark}.cpp ${sources} ${headers})
        target_link_libraries(${benchmark} clSPARSE)
        target_link_libraries(${benchmark} ${Boost_LIBRARIES})
        target_link_libraries(${benchmark} ${clBLAS_LIB})

endforeach(benchmark)
