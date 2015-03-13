project(library)

set(lib_name clSPARSE)

find_package(clBLAS)

message(STATUS "clBLAS include ${clBLAS_INCLUDE_DIR}")
message(STATUS "clBLAS lib ${clBLAS_LIB}")

include_directories(${clBLAS_INCLUDE_DIR})

# loading kernels
file(GLOB kernels "src/kernels/*.cl")


set(ocl_kernels_file_name src/kernels/program_sources.h)
add_custom_command(
   OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/${ocl_kernels_file_name}"
   COMMAND ${CMAKE_COMMAND}
                -DCL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/src/kernels"
                -DOUTPUT="${CMAKE_CURRENT_SOURCE_DIR}/${ocl_kernels_file_name}"
                -P "${CMAKE_SOURCE_DIR}/cmake_modules/cl2c.cmake"
   DEPENDS ${kernels} "${CMAKE_SOURCE_DIR}/cmake_modules/cl2c.cmake"
)

if( NOT WIN32 )
    set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} -std=c99)
    add_definitions(-pedantic)
endif( )

message(STATUS "RELEASE C FLAGS ${CMAKE_C_FLAGS_RELEASE}")
message(STATUS "DEBUG C FLAGS ${CMAKE_C_FLAGS_DEBUG}")
message(STATUS "C FLAGS ${CMAKE_C_FLAGS}")

#make kernels visible in the project
#include_directories(./src/kernels)

include_directories(include) #for include during build time
include_directories(src)
set(LIB_SRC
src/clsparse_init.c
src/internal/hdl_list.c
src/internal/clsparse_internal.c
src/internal/clsparse_control.c
src/simple/clsparse_simple.c
src/spmv/clsparse_csrmv.c
)

set(LIB_HEADERS
include/clSPARSE.h
src/clSPARSE.version.h
src/internal/hdl_list.h
src/internal/clsparse_internal.h
src/internal/clsparse_control.h
src/internal/clsparse_sources.h
src/internal/clsparse_validate.h
${ocl_kernels_file_name}
)

list(APPEND LIB_HEADERS  ${kernels} ${ocl_kernels_file_name})

add_library(${lib_name} ${LIB_SRC} ${LIB_HEADERS} ${OPENCL_INCLUDE_DIRS})

target_link_libraries(${lib_name} ${OPENCL_LIBRARIES} ${clBLAS_LIB})
