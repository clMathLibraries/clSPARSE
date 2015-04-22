#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_IMPL_HPP_
#define _CLSPARSE_CSRMV_VECTOR_IMPL_HPP_


#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
#include "include/clSPARSE_2x.hpp"
#else
#include "include/clSPARSE_1x.hpp"
#endif


clsparseStatus
csrmv_a1b0 (const clsparseCsrMatrixPrivate* pMatx,
            const clsparseVectorPrivate* pX,
            clsparseVectorPrivate* pY,
            const std::string& params,
            const cl_uint group_size,
            const cl_uint subwave_size,
            clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_alpha1_beta0",
                                         "csrmv_alpha1_beta0",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->m
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX->values << pX->offset()
             << pY->values << pX->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMatx->m;


    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus
csrmv_b0 (const clsparseScalarPrivate* pAlpha,
          const clsparseCsrMatrixPrivate* pMatx,
          const clsparseVectorPrivate*  pX,
          clsparseVectorPrivate* pY,
          const std::string& params,
          const cl_uint group_size,
          const cl_uint subwave_size,
          clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_beta0",
                                         "csrmv_beta0",
                                         params);
    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->m
             << pAlpha->value << pAlpha->offset()
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX->values << pX->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMatx->m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

clsparseStatus
csrmv_a1 (const clsparseCsrMatrixPrivate* pMat,
          const clsparseVectorPrivate* pX,
          const clsparseScalarPrivate* pBeta,
          clsparseVectorPrivate* pY,
          const std::string& params,
          const cl_uint group_size,
          const cl_uint subwave_size,
          clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_alpha1",
                                         "csrmv_alpha1",
                                         params);
    KernelWrap kWrapper(kernel);

    kWrapper << pMat->m
             << pMat->rowOffsets
             << pMat->colIndices
             << pMat->values
             << pX->values << pX->offset()
             << pBeta->value << pBeta->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMat->m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);

    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

clsparseStatus
csrmv_b1 (const clsparseScalarPrivate* pAlpha,
          const clsparseCsrMatrixPrivate* pMatx,
          const clsparseVectorPrivate* pX,
          clsparseVectorPrivate* pY,
          const std::string& params,
          const cl_uint group_size,
          const cl_uint subwave_size,
          clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_beta1",
                                         "csrmv_beta1",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->m
             << pAlpha->value << pAlpha->offset()
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX->values << pX->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMatx->m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);

    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    return clsparseSuccess;
}

clsparseStatus
csrmv (const clsparseScalarPrivate* pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const clsparseVectorPrivate* pX,
       const clsparseScalarPrivate* pBeta,
       clsparseVectorPrivate* pY,
       const std::string& params,
       const cl_uint group_size,
       const cl_uint subwave_size,
       clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_general",
                                         "csrmv_general",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->m
             << pAlpha->value << pAlpha->offset()
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX->values << pX->offset()
             << pBeta->value << pBeta->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMatx->m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}
#endif //_CLSPARSE_CSRMV_VECTOR_IMPL_HPP_
