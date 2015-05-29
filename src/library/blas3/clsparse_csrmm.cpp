#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "clsparse_csrmm.hpp"

clsparseStatus
clsparseScsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* matA,
const cldenseMatrix* matB,
const clsparseScalar* beta,
cldenseMatrix* matC,
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
    const clsparseCsrMatrixPrivate* pCsrMatx = static_cast<const clsparseCsrMatrixPrivate*>( matA );
    const cldenseMatrixPrivate* pX = static_cast<const cldenseMatrixPrivate*>( matB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pY = static_cast<cldenseMatrixPrivate*>( matC );


    //return csrmv<cl_float>( pAlpha, pCsrMatx, pX, pBeta, pY, control );
    return clsparseSuccess;
}

clsparseStatus
clsparseDcsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* matA,
const cldenseMatrix* matB,
const clsparseScalar* beta,
cldenseMatrix* matC,
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
    const clsparseCsrMatrixPrivate* pCsrMatx = static_cast<const clsparseCsrMatrixPrivate*>( matA );
    const cldenseMatrixPrivate* pX = static_cast<const cldenseMatrixPrivate*>( matB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pY = static_cast<cldenseMatrixPrivate*>( matC );

    // return csrmv<cl_double>( pAlpha, pCsrMatx, pX, pBeta, pY, control );
    return clsparseSuccess;

}
