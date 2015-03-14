message( STATUS "Configuring clBLAS SuperBuild..." )
include( ExternalProject )

set( ext.clBLAS_Tag "develop" CACHE STRING "clBLAS tag to download" )
mark_as_advanced( ext.clBLAS_Tag )

message( STATUS "ext.clBLAS_Tag: " ${ext.clBLAS_Tag} )

set( ext.clBLAS.cmake_common_args -DBOOST_ROOT=${BOOST_ROOT} -DGTEST_ROOT=${GMOCK_ROOT} )

if( UNIX )
  # Add build thread in addition to the number of cores that we have
  include( ProcessorCount )
  ProcessorCount( Cores )
  message( STATUS "ExternalclBLAS detected ( " ${Cores} " ) cores to build clBLAS with" )

  set( clBLAS.Make "make" )
  if( NOT Cores EQUAL 0 )
    math( EXPR Cores "${Cores} + 1 " )
    list( APPEND clBLAS.Make -j ${Cores} )
  else( )
    # If we could not detect # of cores, assume 1 core and add an additional build thread
    list( APPEND clBLAS.Make -j 2 )
  endif( )
else( )
  set( clBLAS.Make ${CMAKE_COMMAND} --build <BINARY_DIR> )
endif( )

# message( STATUS "External.clBLAS OPENCL_ROOT= " ${OPENCL_ROOT} )
# message( STATUS "External.clBLAS BOOST_ROOT= " ${BOOST_ROOT} )
# message( STATUS "External.clBLAS GTEST_ROOT= " ${GMOCK_ROOT} )

# This dependency requires OpenCL to compile
# A standard FindOpenCL.cmake module ships with cmake 3.1, buy we supply our own until 3.1 becomes more pervasive
find_package( OpenCL REQUIRED )

# The reason I specify CONFIGURE_COMMAND is because the root CMakeFiles.txt is in the src subdirectory; otherwise cmake does not find it
# Since I override CONFIGURE_COMMAND, I have to specify the entire configure line, including -G target and all args; using CMAKE_ARGS does not work
ExternalProject_Add(
  clMATH.clblas
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/Externals/clBLAS
  GIT_REPOSITORY http://www.github.com/clMathLibraries/clBLAS.git
  GIT_TAG ${ext.clBLAS_Tag}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package -DACML_ROOT=${ACML_ROOT} ${ext.clBLAS.cmake_common_args} <SOURCE_DIR>/src
  BUILD_COMMAND ${clBLAS.Make}
  #  LOG_CONFIGURE 1
)

add_dependencies( clMATH.clblas Boost gMock ACML )

set_property( TARGET clMATH.clblas PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( clMATH.clblas binary_dir )
set( CLMATH_BLAS_ROOT ${binary_dir}/package )
