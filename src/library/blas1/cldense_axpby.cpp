#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

clsparseStatus
axpby(cl_ulong size,
      clsparseVectorPrivate* pY,
      const clsparseScalarPrivate* pAlpha,
      const clsparseVectorPrivate* pX,
      const clsparseScalarPrivate* pBeta,
      const std::string& params,
      const cl_uint group_size,
      const clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue, "blas1", "axpby",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pY->values
             << pY->offset()
             << pAlpha->value
             << pAlpha->offset()
             << pX->values
             << pX->offset()
             << pBeta->value
             << pBeta->offset();

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
cldenseSaxpby(clsparseVector *y,
               const clsparseScalar *alpha,
               const clsparseVector *x,
               const clsparseScalar *beta,
               const clsparseControl control)
{

    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );

    const int group_size = 256; // this or higher? control->max_wg_size?


    //is it necessary? Maybe run the kernel nevertheless those values?
//    clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
//    cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

//    clMemRAII<cl_float> rBeta (control->queue(), pBeta->value);
//    cl_float* fBeta = rBeta.clMapMem( CL_TRUE, CL_MAP_READ, pBeta->offset(), 1);

    //nothing to do
    //if (*fAlpha == 0) return clsparseSuccess;

    cl_ulong y_size = pY->n - pY->offValues;
    cl_ulong x_size = pX->n - pX->offValues;

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string( group_size );

    return axpby(size, pY, pAlpha, pX, pBeta, params, group_size, control);
}

clsparseStatus
cldenseDaxpby(clsparseVector *y,
               const clsparseScalar *alpha,
               const clsparseVector *x,
               const clsparseScalar *beta,
               const clsparseControl control)
{
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );

    const int group_size = 256; // this or higher? control->max_wg_size?

//    clMemRAII<cl_double> rAlpha (control->queue(), pAlpha->value);
//    cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

//    clMemRAII<cl_double> rBeta (control->queue(), pBeta->value);
//    cl_double* fBeta = rBeta.clMapMem( CL_TRUE, CL_MAP_READ, pBeta->offset(), 1);


    cl_ulong y_size = pY->n - pY->offValues;
    cl_ulong x_size = pX->n - pX->offValues;

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;

    //TODO: validate object sizes;


    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
            + " -DWG_SIZE=" + std::to_string( group_size );

    return axpby(size, pY, pAlpha, pX, pBeta, params, group_size, control);
}
