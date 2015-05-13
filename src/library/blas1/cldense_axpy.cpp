#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

clsparseStatus
axpy(cl_ulong size,
     clsparseVectorPrivate* pY,
     const clsparseScalarPrivate* pAlpha,
     const clsparseVectorPrivate* pX,
     const std::string& params,
     const cl_uint group_size,
     const clsparseControl control)
{
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

    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    const int group_size = 256; // this or higher? control->max_wg_size?

     clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
     cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->n - pY->offset();
     cl_ulong x_size = pX->n - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;


    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string( group_size );

    return axpy(size, pY, pAlpha, pX, params, group_size, control);
}

clsparseStatus
cldenseDaxpy(clsparseVector *y,
              const clsparseScalar *alpha, const clsparseVector *x,
              const clsparseControl control)
{
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    const int group_size = 256; // this or higher? control->max_wg_size?

     clMemRAII<cl_double> rAlpha (control->queue(), pAlpha->value);
     cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->n - pY->offset();
     cl_ulong x_size = pX->n - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;

     const std::string params = std::string()
             + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
             + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
             + " -DWG_SIZE=" + std::to_string( group_size );

    return axpy(size, pY, pAlpha, pX, params, group_size, control);
}
