#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"

template<typename T>
clsparseStatus
axpy(cl_ulong size,
     clsparseVectorPrivate* pY,
     const clsparseScalarPrivate* pAlpha,
     const clsparseVectorPrivate* pX,
     const clsparseControl control)
{
    const int group_size = 256; // this or higher? control->max_wg_size?

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string( group_size );

    cl::Kernel kernel = KernelCache::get(control->queue, "blas1", "axpy",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pY->values
             << pY->offset()
             << pAlpha->value
             << pAlpha->offset()
             << pX->values
             << pX->offset();

    int blocksNum = (size + group_size - 1) / group_size;
    int globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global (globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

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
