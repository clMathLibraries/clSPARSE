#include <string>
#include <algorithm>

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/computeRowBlocks.hpp"

template<typename FloatType>
struct Coordinate
{
    int x;
    int y;
    FloatType val;
};

template<typename FloatType>
bool CoordinateCompare( const Coordinate<FloatType> &c1, const Coordinate<FloatType> &c2 )
{
    if( c1.x != c2.x )
        return ( c1.x < c2.x );
    else
        return ( c1.y < c2.y );
}

clsparseStatus
coo2csr_transform( cl_float* fCooValues, cl_int* iCooColIndices, cl_int* iCooRowIndices, cl_int nNZ,
cl_float* fCsrValues, cl_int* iCsrColIndices, cl_int* iCsrRowOffsets )
{
    // TODO:  This coalescing is inefficient, but should be done on the GPU anyways
    std::vector< Coordinate< float > > coords( nNZ );

    int index = 0;
    for( cl_int index = 0; index < nNZ; ++index )
    {
        coords[ index ].val = fCooValues[ index ];
        coords[ index ].x = iCooRowIndices[ index ];
        coords[ index ].y = iCooColIndices[ index ];
    }
    std::sort( coords.begin( ), coords.end( ), CoordinateCompare< float > );

    int current_row = 1;
    iCsrRowOffsets[ 0 ] = 0;
    for( int i = 0; i < nNZ; i++ )
    {
        iCsrColIndices[ i ] = coords[ i ].y;
        fCsrValues[ i ] = coords[ i ].val;

        if( coords[ i ].x >= current_row )
            iCsrRowOffsets[ current_row++ ] = i;
    }
    iCsrRowOffsets[ current_row ] = nNZ;

    return clsparseSuccess;
}

clsparseStatus
clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control )
{
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );

    // This allocates up front the maximum size of rowBlocks at start; likely not all the memory is used but
    // this is the fastest 
    pCsrMatx->rowBlockSize = 3 * ( pCsrMatx->nnz / BLKSIZE ) + 1;

    return clsparseSuccess;
}

clsparseStatus
clsparseCsrMetaCompute( clsparseCsrMatrix* csrMatx, clsparseControl control )
{
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );

    // Check to ensure nRows can fit in 32 bits
    if( static_cast<cl_ulong>( pCsrMatx->m ) > static_cast<cl_ulong>( pow( 2, ( 64 - ROW_BITS ) ) ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present ((64-WG_BITS) bits) !" );
        return clsparseOutOfResources;
    }

    clMemRAII< cl_int > rCsrRowOffsets( control->queue( ), pCsrMatx->rowOffsets );
    cl_int* rowDelimiters = rCsrRowOffsets.clMapMem( CL_TRUE, CL_MAP_READ, pCsrMatx->rowOffOffset( ), pCsrMatx->m + 1 );

    clMemRAII< cl_ulong > rRowBlocks( control->queue( ), pCsrMatx->rowBlocks );
    cl_ulong* ulCsrRowBlocks = rRowBlocks.clMapMem( CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, pCsrMatx->rowBlocksOffset( ), pCsrMatx->rowBlockSize );

    ComputeRowBlocks( ulCsrRowBlocks, pCsrMatx->rowBlockSize, rowDelimiters, pCsrMatx->m, BLKSIZE );

    return clsparseSuccess;
}

// Converts a sparse matrix in COO compressed format into CSR compressed format
// Pre-condition: The CL device memory for CSR values, colIndices, rowOffsets has to be allocated prior to entering this routine
// and the offset variables for cl1.2 set
clsparseStatus
clsparseScoo2csr( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx, clsparseControl control )
{
    if( !clsparseInitialized )
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if( control == nullptr )
    {
        return clsparseInvalidControlObject;
    }

    const clsparseCooMatrixPrivate* pCooMatx = static_cast<const clsparseCooMatrixPrivate*>( cooMatx );
    clsparseCsrMatrixPrivate* pCsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );
    pCsrMatx->m = pCooMatx->m;
    pCsrMatx->n = pCooMatx->n;
    pCsrMatx->nnz = pCooMatx->nnz;

    clMemRAII< cl_float > rCooValues( control->queue( ), pCooMatx->values );
    clMemRAII< cl_int > rCooColIndices( control->queue( ), pCooMatx->colIndices );
    clMemRAII< cl_int > rCooRowIndices( control->queue( ), pCooMatx->rowIndices );
    clMemRAII< cl_float > rCsrValues( control->queue( ), pCsrMatx->values );
    clMemRAII< cl_int > rCsrColIndices( control->queue( ), pCsrMatx->colIndices );
    clMemRAII< cl_int > rCsrRowOffsets( control->queue( ), pCsrMatx->rowOffsets );

    cl_float* fCooValues = rCooValues.clMapMem( CL_TRUE, CL_MAP_READ, pCooMatx->valOffset( ), pCooMatx->nnz );
    cl_int* iCooColIndices = rCooColIndices.clMapMem( CL_TRUE, CL_MAP_READ, pCooMatx->colIndOffset( ), pCooMatx->nnz );
    cl_int* iCooRowIndices = rCooRowIndices.clMapMem( CL_TRUE, CL_MAP_READ, pCooMatx->rowOffOffset( ), pCooMatx->nnz );

    cl_float* fCsrValues = rCsrValues.clMapMem( CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, pCsrMatx->valOffset( ), pCsrMatx->nnz );
    cl_int* iCsrColIndices = rCsrColIndices.clMapMem( CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, pCsrMatx->colIndOffset( ), pCsrMatx->nnz );
    cl_int* iCsrRowOffsets = rCsrRowOffsets.clMapMem( CL_TRUE, CL_MAP_WRITE_INVALIDATE_REGION, pCsrMatx->rowOffOffset( ), pCsrMatx->m + 1 );

    coo2csr_transform( fCooValues, iCooColIndices, iCooRowIndices, pCooMatx->nnz, fCsrValues, iCsrColIndices, iCsrRowOffsets );

    return clsparseSuccess;
}

clsparseStatus
clsparseDcoo2csr( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx, clsparseControl control )
{
    if( !clsparseInitialized )
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if( control == nullptr )
    {
        return clsparseInvalidControlObject;
    }

    const clsparseCooMatrixPrivate* pcooMatx = static_cast<const clsparseCooMatrixPrivate*>( cooMatx );
    clsparseCsrMatrixPrivate* pcsrMatx = static_cast<clsparseCsrMatrixPrivate*>( csrMatx );

    return clsparseSuccess;
}

