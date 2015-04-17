#include <stdlib.h>
#include <stdio.h>

#include "clSPARSE.h"
#include "clSPARSE.version.h"
#include "internal/clsparse_internal.hpp"

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE_2x.hpp"
#else
    #include "include/clSPARSE_1x.hpp"
#endif

#include <clAmdBlas.h>

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
    clsparseScalarPrivate* pScalar = static_cast<clsparseScalarPrivate*>( scalar );
    pScalar->clear( );

    return clsparseSuccess;
};

clsparseStatus
clsparseInitVector( clsparseVector* vec )
{
    clsparseVectorPrivate* pVec = static_cast<clsparseVectorPrivate*>( vec );
    pVec->clear( );

    return clsparseSuccess;
};

clsparseStatus
clsparseInitCooMatrix( clsparseCooMatrix* cooMatx )
{
    clsparseCooMatrixPrivate* pCooMatx = static_cast<clsparseCooMatrixPrivate*>( cooMatx );
    pCooMatx->clear( );

    return clsparseSuccess;
};

clsparseStatus
clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx )
{
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );
    pCsrMatx->clear( );

    return clsparseSuccess;
};

clsparseStatus
clsparseInitDenseMatrix (clsparseDenseMatrix *denseMatx)
{
    clsparseDenseMatrixPrivate* pDenseMatx = static_cast<clsparseDenseMatrixPrivate*> ( denseMatx );
    pDenseMatx->clear();

    return clsparseSuccess;
}

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
