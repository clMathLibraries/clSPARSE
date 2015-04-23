#include "clSPARSE.h"

#include "internal/clsparse_control.hpp"
#include "spmv/csrmv_vector/csrmv_vector.hpp"

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE_2x.hpp"
#else
    #include "include/clSPARSE_1x.hpp"
#endif



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
    const clsparseCsrMatrixPrivate* pMatx = static_cast<const clsparseCsrMatrixPrivate*>( matx );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*>( y );

    if( matx->rowBlocks == nullptr )
    {
        return clsparseScsrmv_vector(pAlpha, pMatx, pX, pBeta, pY, control);
    }
    else
    {
        // Call adaptive CSR kernels
        return clsparseNotImplemented;
    }

    return clsparseNotImplemented;
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

    return clsparseNotImplemented;
}
