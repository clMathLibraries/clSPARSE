#include "include/clSPARSE-private.hpp"
#include "cldense-axpby.hpp"

clsparseStatus
cldenseSaxpby(cldenseVector *y,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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


    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );

    //is it necessary? Maybe run the kernel nevertheless those values?
//    clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
//    cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

//    clMemRAII<cl_float> rBeta (control->queue(), pBeta->value);
//    cl_float* fBeta = rBeta.clMapMem( CL_TRUE, CL_MAP_READ, pBeta->offset(), 1);

    //nothing to do
    //if (*fAlpha == 0) return clsparseSuccess;

    cl_ulong y_size = pY->n - pY->offset();
    cl_ulong x_size = pX->n - pX->offset();

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;



    return axpby<cl_float>(size, pY, pAlpha, pX, pBeta, control);
}

clsparseStatus
cldenseDaxpby(cldenseVector *y,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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

    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );


    cl_ulong y_size = pY->n - pY->offset();
    cl_ulong x_size = pX->n - pX->offset();

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;


    return axpby<cl_double>(size, pY, pAlpha, pX, pBeta, control);
}
