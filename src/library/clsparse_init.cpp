#include "clSPARSE.h"
#include "clSPARSE.version.h"
#include "internal/clsparse_internal.hpp"

#include <clAmdBlas.h>
#include <stdlib.h>
#include <stdio.h>

int clsparseInitialized = 0;

clsparseStatus
clsparseGetVersion( cl_uint *major, cl_uint *minor, cl_uint *patch, cl_uint *tweak )
{
    *major = clsparseVersionMajor;
    *minor = clsparseVersionMinor;
    *patch = clsparseVersionPatch;
    *tweak = clsparseVersionTweak;

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

// Convenience sparse matrix construction functions
clsparseStatus
clsparseInitScalar( clsparseScalar* scalar )
{
    scalar->value = nullptr;

    scalar->offValue = 0;

    return clsparseSuccess;
};

clsparseStatus
clsparseInitVector( clsparseVector* vec )
{
    vec->n = 0;

    vec->values = nullptr;
    vec->offValues = 0;

    return clsparseSuccess;
};

clsparseStatus
clsparseInitCooMatrix( clsparseCooMatrix* cooMatx )
{
    cooMatx->m = 0;
    cooMatx->n = 0;
    cooMatx->nnz = 0;

    cooMatx->values = nullptr;
    cooMatx->colIndices = nullptr;
    cooMatx->rowIndices = nullptr;
    cooMatx->offValues = 0;
    cooMatx->offColInd = 0;
    cooMatx->offRowInd = 0;

    return clsparseSuccess;
};

clsparseStatus
clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx )
{
    csrMatx->m = 0;
    csrMatx->n = 0;
    csrMatx->nnz = 0;

    csrMatx->values = nullptr;
    csrMatx->colIndices = nullptr;
    csrMatx->rowOffsets = nullptr;
    csrMatx->rowBlocks = nullptr;
    csrMatx->offValues = 0;
    csrMatx->offColInd = 0;
    csrMatx->offRowBlocks = 0;
    csrMatx->offRowOff = 0;

    return clsparseSuccess;
};

CLSPARSE_EXPORT clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath )
{
    // There should be logic here to read matrix market files
    // First, it reads data from file into CPU buffer
    // Then, it transfers the data from CPU buffer to GPU buffers
    // Ponder:  should we suppose the usecase of allocating 1 cl_mem buffer (values == colIndices == rowIndices )
    // with all data addressed with offsets ( offValues != offColInd != offRowInd != 0 )?
    return clsparseInitCooMatrix( cooMatx );
}

CLSPARSE_EXPORT clsparseStatus
clsparseCoo2Csr( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx )
{
    // There should be logic to convert coo to csr here
    // This assumes that the data is sitting in GPU buffers, and converts data in place
    // documentation says that this routins is asynchronous; implying gpu execution
    // This routine IS RESPONSIBLE FOR creating the rowBlocks data for CSR adaptive algorithms
    // Ponder:  should we suppose the usecase of allocating 1 cl_mem buffer (values == colIndices == rowIndices )
    // with all data addressed with offsets within that buffer ( offValues != offColInd != offRowInd != 0 )?
    return clsparseInitCsrMatrix( csrMatx );
}
