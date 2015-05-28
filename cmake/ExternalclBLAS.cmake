message( STATUS "Configuring clBLAS SuperBuild..." )
include( ExternalProject )

set( ext.clBLAS_Tag "master" CACHE STRING "clBLAS tag to download" )
mark_as_advanced( ext.clBLAS_Tag )

message( STATUS "ext.clBLAS_Tag: " ${ext.clBLAS_Tag} )

set( ext.clBLAS.BUILD.options -DBUILD_RUNTIME=ON -DBUILD_CLIENT=OFF -DBUILD_KTEST=OFF -DBUILD_PERFORMANCE=OFF -DBUILD_SAMPLE=OFF -DBUILD_TEST=OFF )

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

# clBLAS requires an OpenCL SDK to compile the library
# There are multiple vendors, the AMD SDK requires the installation through an installer for windows
# and the download links are obscured through a click-through.  Let clBLAS handle finding OpenCL
# find_package( OpenCL )

# The reason I specify CONFIGURE_COMMAND is because the root CMakeFiles.txt is in the src subdirectory; otherwise cmake does not find it
# Since I override CONFIGURE_COMMAND, I have to specify the entire configure line, including -G target and all args; using CMAKE_ARGS does not work
ExternalProject_Add(
  clMATH.clblas
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/Externals/clBLAS
  GIT_REPOSITORY http://www.github.com/clMathLibraries/clBLAS.git
  GIT_TAG ${ext.clBLAS_Tag}
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -G${CMAKE_GENERATOR} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>/package ${ext.clBLAS.BUILD.options} <SOURCE_DIR>/src
  BUILD_COMMAND ${clBLAS.Make}
  #  LOG_CONFIGURE 1
)

set_property( TARGET clMATH.clblas PROPERTY FOLDER "Externals")

ExternalProject_Get_Property( clMATH.clblas install_dir )
set( CLMATH_BLAS_ROOT ${install_dir}/package )
