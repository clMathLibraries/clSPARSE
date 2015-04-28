#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "spmv/csrmv_adaptive/csrmv_adaptive.hpp"
#include "spmv/csrmv_vector/csrmv_vector.hpp"

//Dummy implementation of new interface;
clsparseStatus
clsparseScsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            const clsparseScalar* beta,
            clsparseVector* y,
            const clsparseControl control )
{
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pCsrMatx = static_cast<const clsparseCsrMatrixPrivate*>( matx );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*>( y );

    if( (pCsrMatx->rowBlocks == nullptr) && (pCsrMatx->rowBlockSize == 0) )
    {
        return clsparseScsrmv_vector( pAlpha, pCsrMatx, pX, pBeta, pY, control );
    }
    else
    {
        if( ( pCsrMatx->rowBlocks == nullptr ) || ( pCsrMatx->rowBlockSize == 0 ) )
        {
            // rowBlockSize varible is not zero but no pointer
            return clsparseStructInvalid;
        }

        // Call adaptive CSR kernels
        return clsparseScsrmv_adaptive( *pAlpha, *pCsrMatx, *pX, *pBeta, *pY, control );
    }

    return clsparseSuccess;
}

clsparseStatus
clsparseDcsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            const clsparseScalar* beta,
            clsparseVector* y,
            const clsparseControl control )
{
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pMatx = static_cast<const clsparseCsrMatrixPrivate*>( matx );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*>( y );

    if( matx->rowBlocks == nullptr )
    {
        return clsparseDcsrmv_vector(pAlpha, pMatx, pX, pBeta, pY, control);
    }
    else
    {
        // Call adaptive CSR kernels
        return clsparseNotImplemented;
    }

    return clsparseSuccess;
}
