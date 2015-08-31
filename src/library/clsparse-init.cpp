/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#include <stdlib.h>
#include <stdio.h>

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-internal.hpp"
#include "clSPARSE-version.h"

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
clsparseInitVector( cldenseVector* vec )
{
    cldenseVectorPrivate* pVec = static_cast<cldenseVectorPrivate*>( vec );
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
cldenseInitMatrix( cldenseMatrix* denseMatx )
{
    cldenseMatrixPrivate* pDenseMatx = static_cast<cldenseMatrixPrivate*> ( denseMatx );
    pDenseMatx->clear();

    return clsparseSuccess;
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
