#include "include/clSPARSE-private.hpp"
#include "cldense_dot.hpp"
clsparseStatus
cldenseSdot (clsparseScalar* r,
             const clsparseVector* x,
             const clsparseVector* y,
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
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseVectorPrivate* pY = static_cast<const clsparseVectorPrivate*> ( y );

    return dot<cl_float>(pR, pX, pY, control);
}

clsparseStatus
cldenseDdot (clsparseScalar* r,
             const clsparseVector* x,
             const clsparseVector* y,
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
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseVectorPrivate* pY = static_cast<const clsparseVectorPrivate*> ( y );

    return dot<cl_double>(pDot, pX, pY, control);
}
