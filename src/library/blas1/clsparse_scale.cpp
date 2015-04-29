#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include <clBLAS.h>


//TODO:: add offset to the scale kernel
clsparseStatus
scale(clsparseVectorPrivate* pVector,
      const clsparseScalarPrivate* pAlpha,
      const std::string& params,
      const cl_uint group_size,
      clsparseControl control)
{

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "blas1", "scale",
                                         params);
    KernelWrap kWrapper(kernel);


    kWrapper << (cl_ulong)pVector->n
             << pVector->values
             << pVector->offset()
             << pAlpha->value
             << pAlpha->offset();

    int blocksNum = (pVector->n + group_size - 1) / group_size;
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
clsparseSscale (clsparseVector* y,
                const clsparseScalar* alpha,
                const clsparseControl control)
{

    clMemRAII<cl_float> rAlpha (control->queue(), alpha->value);
    cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, alpha->offValue, 1);


#if( BUILD_CLVERSION < 200 )


    clblasStatus status =
            clblasSscal(y->n, *fAlpha, y->values, y->offValues,
                1, 1,
                &control->queue(),
                control->event_wait_list.size(),
                &(control->event_wait_list.front())(),
                &control->event( ));

    if(status != clblasSuccess)
        return clsparseInvalidKernelExecution;
    else
        return clsparseSuccess;
#else
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );

    cl_float pattern = 0.0f;

    if (*fAlpha == 0)
    {
        cl_int status =
                clEnqueueFillBuffer(control->queue(), pY->values,
                                    &pattern, sizeof(cl_float), 0,
                                    sizeof(cl_float) * pY->n,
                                    control->event_wait_list.size(),
                                    &(control->event_wait_list.front())(),
                                    &control->event( ));
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    return scale(pY, pAlpha, params, group_size, control);

#endif

}

clsparseStatus
clsparseDscale (clsparseVector* y,
                const clsparseScalar* alpha,
                const clsparseControl control)
{
    clMemRAII<cl_double> rAlpha (control->queue(), alpha->value);
    cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, alpha->offValue, 1);

#if( BUILD_CLVERSION < 200 )

    clblasStatus status =
            clblasDscal(y->n, *fAlpha, y->values, y->offValues,
                1, 1,
                &control->queue(),
                control->event_wait_list.size(),
                &(control->event_wait_list.front())(),
                &control->event( ));

    if(status != clblasSuccess)
        return clsparseInvalidKernelExecution;
    else
        return clsparseSuccess;
#else
    clsparseVectorPrivate* pY = static_cast<clsparseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );

    cl_double pattern = 0.0f;

    if (*fAlpha == 0)
    {
        cl_int status =
                clEnqueueFillBuffer(control->queue(), pY->values,
                                    &pattern, sizeof(cl_double), 0,
                                    sizeof(cl_double) * pY->n,
                                    control->event_wait_list.size(),
                                    &(control->event_wait_list.front())(),
                                    &control->event( ));
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<cl_double>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    return scale(pY, pAlpha, params, group_size, control);


#endif

}
