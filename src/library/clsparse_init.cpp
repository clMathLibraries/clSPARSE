#include "clSPARSE.h"
#include "clSPARSE.version.h"
#include "internal/clsparse_internal.hpp"

#include <clAmdBlas.h>
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

    //program sources will keep kernel sources therefore false
    program_sources = hdl_create(false);

    //kernel cache will keep the clKernel object so true
    kernel_cache = hdl_create(true);

    createSourcesMap();

    clblasSetup();

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
    hdl_destroy_with_func(&kernel_cache, (free_clfunc_t)(&clReleaseKernel));

    clblasTeardown();

    clsparseInitialized = 0;
    return clsparseSuccess;
}
