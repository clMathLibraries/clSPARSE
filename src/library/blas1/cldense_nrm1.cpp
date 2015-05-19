#include "include/clSPARSE-private.hpp"
#include "cldense_nrm1.hpp"

clsparseStatus
cldenseSnrm1(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm1<cl_float>(pS, pX, control);

}

clsparseStatus
cldenseDnrm1(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm1<cl_double>(pS, pX, control);
}
