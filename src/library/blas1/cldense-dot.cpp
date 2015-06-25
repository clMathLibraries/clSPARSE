#include "include/clSPARSE-private.hpp"
#include "cldense-dot.hpp"
clsparseStatus
cldenseSdot (clsparseScalar* r,
             const cldenseVector* x,
             const cldenseVector* y,
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


    clsparseScalarPrivate* pR = static_cast<clsparseScalarPrivate*>( r );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate* pY = static_cast<const cldenseVectorPrivate*> ( y );

    return dot<cl_float>(pR, pX, pY, control);
}

clsparseStatus
cldenseDdot (clsparseScalar* r,
             const cldenseVector* x,
             const cldenseVector* y,
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


    clsparseScalarPrivate* pDot = static_cast<clsparseScalarPrivate*>( r );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate* pY = static_cast<const cldenseVectorPrivate*> ( y );

    return dot<cl_double>(pDot, pX, pY, control);
}
