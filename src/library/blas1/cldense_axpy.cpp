#include "include/clSPARSE-private.hpp"
#include "blas1/cldense_axpy.hpp"


clsparseStatus
cldenseSaxpy(clsparseVector *y,
              const clsparseScalar *alpha,
              const clsparseVector *x,
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

     clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
     cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->n - pY->offset();
     cl_ulong x_size = pX->n - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;

    return axpy<cl_float>(size, pY, pAlpha, pX, control);
}

clsparseStatus
cldenseDaxpy(clsparseVector *y,
              const clsparseScalar *alpha, const clsparseVector *x,
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


     clMemRAII<cl_double> rAlpha (control->queue(), pAlpha->value);
     cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->n - pY->offset();
     cl_ulong x_size = pX->n - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;


    return axpy<cl_double>(size, pY, pAlpha, pX, control);
}
