# ########################################################################
# Copyright 2015 Advanced Micro Devices, Inc.
# ########################################################################
include( GenerateExportHeader )

# List the names of the files to compile for the external client . . .
set( clsparseTimer.Source
	clsparseTimer.host.cpp
	clsparseTimer.device.cpp
	clsparseTimer.extern.cpp
 )

# Windows only uses dllmain
if( MSVC )
	list( APPEND clsparseTimer.Source dllmain.cpp )
endif( )

set( clsparseTimer.Headers
	${PROJECT_SOURCE_DIR}/include/clSPARSE.h
	${PROJECT_SOURCE_DIR}/include/clsparseTimer.hpp
	${PROJECT_SOURCE_DIR}/include/clsparseTimer.extern.hpp
	clsparseTimer.host.hpp
	clsparseTimer.device.hpp
 )

set( clsparseTimer.Files ${clsparseTimer.Source} ${clsparseTimer.Headers} )

if( CMAKE_COMPILER_IS_GNUCXX OR ( CMAKE_CXX_COMPILER_ID MATCHES "Clang" ) )
  add_definitions( -std=c++11 )
endif( )

# Include standard OpenCL headers
include_directories(
	${OPENCL_INCLUDE_DIRS}
	${PROJECT_BINARY_DIR}/include
	${PROJECT_BINARY_DIR}/library
	${PROJECT_BINARY_DIR}/clsparseTimer
	../include
	)

add_compiler_export_flags( )
add_library( clsparseTimer SHARED ${clsparseTimer.Files} )
set_target_properties( clsparseTimer PROPERTIES VERSION ${clSPARSE_VERSION} )
set_target_properties( clsparseTimer PROPERTIES SOVERSION ${clSPARSE_SOVERSION} )
set_target_properties( clsparseTimer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging" )
target_link_libraries( clsparseTimer ${OPENCL_LIBRARIES} )

# Package that helps me set visibility for function names exported from shared library
GENERATE_EXPORT_HEADER( clsparseTimer )

if( UNIX AND NOT APPLE )
	# This library dependency is brought in by the high precision timer available in linux
	target_link_libraries( clsparseTimer -lrt )
endif( )

# CPack configuration; include the executable into the package
install( TARGETS clsparseTimer
  RUNTIME DESTINATION bin${SUFFIX_BIN}
  LIBRARY DESTINATION lib${SUFFIX_LIB}
  ARCHIVE DESTINATION lib${SUFFIX_LIB}
)

install( FILES
  ${CMAKE_CURRENT_BINARY_DIR}/clsparseTimer_export.h
  DESTINATION
    include
)
