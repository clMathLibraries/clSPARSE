#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "clsparse-csrmv.hpp"

clsparseStatus
clsparseScsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            const clsparseScalar* beta,
            clsparseVector* y,
            const clsparseControl control )
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pCsrMatx = static_cast<const clsparseCsrMatrixPrivate*>( matx );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*>( y );


    return csrmv<cl_float>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
}

clsparseStatus
clsparseDcsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            const clsparseScalar* beta,
            clsparseVector* y,
            const clsparseControl control )
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pCsrMatx = static_cast<const clsparseCsrMatrixPrivate*>( matx );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*>( y );

    return csrmv<cl_double>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
}
