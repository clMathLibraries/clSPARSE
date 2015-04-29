#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

clsparseStatus
axpy(clsparseVectorPrivate* pY,
     const clsparseScalarPrivate* pAlpha,
     const clsparseVectorPrivate* pX,
     const std::string& params,
     const cl_uint group_size,
     const clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue, "blas1", "axpy",
                                         params);

    KernelWrap kWrapper(kernel);

    cl_ulong pXsize = pX->n - pX->offset();

    kWrapper << pXsize
             << pY->values
             << pY->offset()
             << pAlpha->value
             << pAlpha->offset()
             << pX->values
             << pX->offset();

    int blocksNum = (pXsize + group_size - 1) / group_size;
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
clsparseSaxpy(clsparseVector *y,
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

     //pY->n is the main dimmension. W can allow that part of the pY will be updated;
     //kernel will iterate over pX;
    assert((pY->n - pY->offValues) >= (pX->n - pX->offValues));


    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string( group_size );

    return axpy(pY, pAlpha, pX, params, group_size, control);
}

clsparseStatus
clsparseDaxpy(clsparseVector *y,
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

     //TODO: validate object sizes;

     //pY->n is the main dimmension. W can allow that part of the pY will be updated;
     //kernel will iterate over pX;
     assert((pY->n - pY->offValues) >= (pX->n - pX->offValues));

     const std::string params = std::string()
             + " -DSIZE_TYPE=" + OclTypeTraits<cl_uint>::type
             + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
             + " -DWG_SIZE=" + std::to_string( group_size );

    return axpy(pY, pAlpha, pX, params, group_size, control);
}
