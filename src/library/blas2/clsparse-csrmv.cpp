#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "clsparse-csrmv.hpp"

clsparseStatus
clsparseScsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const cldenseVector* x,
            const clsparseScalar* beta,
            cldenseVector* y,
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
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*>( y );


    return csrmv<cl_float>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
}

clsparseStatus
clsparseDcsrmv( const clsparseScalar* alpha,
            const clsparseCsrMatrix* matx,
            const cldenseVector* x,
            const clsparseScalar* beta,
            cldenseVector* y,
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
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*>( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*>( y );

    return csrmv<cl_double>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
}
