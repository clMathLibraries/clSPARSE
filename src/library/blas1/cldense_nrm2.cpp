#include "include/clSPARSE-private.hpp"
#include "cldense_nrm2.hpp"

clsparseStatus
cldenseSnrm2(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm2<cl_float>(pS, pX, control);

}

clsparseStatus
cldenseDnrm2(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm2<cl_double>(pS, pX, control);
}
