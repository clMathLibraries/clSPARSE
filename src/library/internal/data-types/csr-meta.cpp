/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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


#include "csr-meta.hpp"

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"


clsparseStatus
clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control )
{
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );

    // This allocates up front the maximum size of rowBlocks at start; likely not all the memory is used but
    // this is the fastest
    // The formula is 3 * (NNZ / block size) + 2, but we double this because CSR-Adaptive uses the
    // second half of the rowBlocks buffer for global reductions.
    pCsrMatx->rowBlockSize = 6 * ( pCsrMatx->num_nonzeros / BLKSIZE ) + 4;

    return clsparseSuccess;
}

clsparseStatus
clsparseCsrMetaCompute( clsparseCsrMatrix* csrMatx, clsparseControl control )
{
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );

    // Check to ensure nRows can fit in 32 bits
    if( static_cast<cl_ulong>( pCsrMatx->num_rows ) > static_cast<cl_ulong>( pow( 2, ( 64 - ROW_BITS ) ) ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present ((64-WG_BITS) bits) !" );
        return clsparseOutOfResources;
    }

    clMemRAII< cl_int > rCsrRowOffsets( control->queue( ), pCsrMatx->rowOffsets );
    cl_int* rowDelimiters = rCsrRowOffsets.clMapMem( CL_TRUE, CL_MAP_READ, pCsrMatx->rowOffOffset( ), pCsrMatx->num_rows + 1 );

    clMemRAII< cl_ulong > rRowBlocks( control->queue( ), pCsrMatx->rowBlocks );
    cl_ulong* ulCsrRowBlocks = rRowBlocks.clMapMem( CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, pCsrMatx->rowBlocksOffset( ), pCsrMatx->rowBlockSize );

    ComputeRowBlocks( ulCsrRowBlocks, pCsrMatx->rowBlockSize, rowDelimiters, pCsrMatx->num_rows, BLKSIZE, BLOCK_MULTIPLIER, ROWS_FOR_VECTOR, true );

    return clsparseSuccess;
}
