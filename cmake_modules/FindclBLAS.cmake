if(NOT clBLAS_FOUND)

find_path(clBLAS_INCLUDE_DIRS
    NAMES clAmdBlas.h clBLAS.h
    HINTS
        ${clBLAS_ROOT}/include
    PATHS
        /home/jpola/Projects/clBLAS/build
    DOC "clBLAS header file path"
)

    mark_as_advanced(clBLAS_INCLUDE_DIR)

    include(FindPackageHandleStandardArgs)
    find_package_handle_standard_args(clBLAS DEFAULT_MSG clBLAS_INCLUDE_DIRS)

    set(clBLAS_INCLUDE_DIR ${clBLAS_INCLUDE_DIRS})

find_library( clBLAS_LIB
    NAMES clBLAS
    HINTS
         ${clBLAS_ROOT}/lib
    DOC "clBLAS dynamic libraries"
    PATH_SUFFIXES x86_64 x64 64
    PATHS
        /home/jpola/Projects/clBLAS/build
)

endif(NOT clBLAS_FOUND)
