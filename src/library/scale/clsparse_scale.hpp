#pragma once
#ifndef _CLSPARSE_SCALE_HPP_
#define _CLSPARSE_SCALE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include <clBLAS.h>


//Scale kernel for internal use.
//Assumes that parameters are validated!


//TODO:: add offset to the scale kernel
clsparseStatus
scale(cl_int size,
      const clsparseScalarPrivate* pAlpha,
      clsparseVectorPrivate* pVector,
      const std::string& params,
      const cl_uint group_size,
      clsparseControl control)
{

    assert(pVector->n == size);
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "scale", "scale",
                                         params);
    KernelWrap kWrapper(kernel);


    kWrapper << pVector->values
             << pAlpha->value
             << size;

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
clsparseSscale (cl_int size,
                cl_float halpha,      //I don't want to map alpha on host twice;
                const clsparseScalarPrivate* pAlpha,
                clsparseVectorPrivate* pVector,
                clsparseControl control)
{

#if( BUILD_CLVERSION < 200 )

    clblasStatus status =
            clblasSscal(size, halpha, pVector->values, pVector->offset(),
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

    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    return scale(size, pAlpha, pVector, params, group_size, control);

#endif

}

clsparseStatus
clsparseDscale (cl_int size,
                cl_double halpha,      //I don't want to map alpha on host twice;
                const clsparseScalarPrivate* pAlpha,
                clsparseVectorPrivate* pVector,
                clsparseControl control)
{

#if( BUILD_CLVERSION < 200 )

    clblasStatus status =
            clblasDscal(size, halpha, pVector->values, pVector->offset(),
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
    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<cl_double>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    return scale(size, pAlpha, pVector, params, group_size, control);


#endif

}


#endif //_CLSPARSE_SCALE_HPP_
