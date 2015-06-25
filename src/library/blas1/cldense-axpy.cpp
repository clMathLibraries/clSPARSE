#include "include/clSPARSE-private.hpp"
#include "blas1/cldense-axpy.hpp"


clsparseStatus
cldenseSaxpy(cldenseVector *y,
              const clsparseScalar *alpha,
              const cldenseVector *x,
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


    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );

     clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
     cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->num_values - pY->offset();
     cl_ulong x_size = pX->num_values - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;

    return axpy<cl_float>(size, pY, pAlpha, pX, control);
}

clsparseStatus
cldenseDaxpy(cldenseVector *y,
              const clsparseScalar *alpha, const cldenseVector *x,
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

    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );


     clMemRAII<cl_double> rAlpha (control->queue(), pAlpha->value);
     cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->num_values - pY->offset();
     cl_ulong x_size = pX->num_values - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;


    return axpy<cl_double>(size, pY, pAlpha, pX, control);
}
