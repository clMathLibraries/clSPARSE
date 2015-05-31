#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_control.hpp"
#include "clsparse_csrmm.hpp"

clsparseStatus
clsparseScsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* sparseCsrA,
const cldenseMatrix* denseB,
const clsparseScalar* beta,
cldenseMatrix* denseC,
const clsparseControl control )
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

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pSparseCsrA = static_cast<const clsparseCsrMatrixPrivate*>( sparseCsrA );
    const cldenseMatrixPrivate* pDenseB = static_cast<const cldenseMatrixPrivate*>( pDenseB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pDenseC = static_cast<cldenseMatrixPrivate*>( denseC );


    return csrmm< cl_float >( *pAlpha, *pSparseCsrA, *pDenseB, *pBeta, *pDenseC, control );
}

clsparseStatus
clsparseDcsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* sparseCsrA,
const cldenseMatrix* denseB,
const clsparseScalar* beta,
cldenseMatrix* denseC,
const clsparseControl control )
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

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pSparseCsrA = static_cast<const clsparseCsrMatrixPrivate*>( sparseCsrA );
    const cldenseMatrixPrivate* pDenseB = static_cast<const cldenseMatrixPrivate*>( pDenseB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pDenseC = static_cast<cldenseMatrixPrivate*>( denseC );

    return csrmm< cl_double >( *pAlpha, *pSparseCsrA, *pDenseB, *pBeta, *pDenseC, control );

}
