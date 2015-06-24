#include "include/clSPARSE-private.hpp"
#include "cldense-nrm2.hpp"

clsparseStatus
cldenseSnrm2(clsparseScalar* s,
             const cldenseVector* x,
             const clsparseControl control)
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

    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm2<cl_float>(pS, pX, control);

}

clsparseStatus
cldenseDnrm2(clsparseScalar* s,
             const cldenseVector* x,
             const clsparseControl control)
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

    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return Norm2<cl_double>(pS, pX, control);
}
