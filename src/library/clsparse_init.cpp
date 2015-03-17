#include "clSPARSE.h"
#include "clSPARSE.version.h"
#include "internal/clsparse_internal.hpp"

#include <clAmdBlas.h>
#include <stdlib.h>
#include <stdio.h>

int clsparseInitialized = 0;

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

    clblasTeardown();

    clsparseInitialized = 0;
    return clsparseSuccess;
}
