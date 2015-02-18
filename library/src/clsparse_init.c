#include "clSPARSE.h"
#include "clSPARSE.version.h"

#include "internal/clsparse_internal.h"
#include <stdlib.h>
#include <stdio.h>

clsparseStatus
clsparseGetVersion(cl_uint *major, cl_uint *minor, cl_uint *patch)
{
    *major = clsparseVersionMajor;
    *minor = clsparseVersionMinor;
    *patch = clsparseVersionPatch;

    return clsparseSuccess;
}


clsparseStatus
clsparseSetup(void)
{
    if(clsparseInitialized)
    {
        return clsparseSuccess;
    }

    program_sources = hdl_create(false);
    kernel_cache = hdl_create(true);

    clsparseInitialized = 1;
    return clsparseSuccess;
}

clsparseStatus
clsparseTeardown(void)
{
    if(!clsparseInitialized)
    {
        return clsparseSuccess;
    }

    hdl_destroy(&program_sources);
    hdl_destroy_with_func(&kernel_cache, &clReleaseKernel);

    clsparseInitialized = 0;
    return clsparseSuccess;
}
