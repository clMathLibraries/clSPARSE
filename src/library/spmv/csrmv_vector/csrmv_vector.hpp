#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_HPP_
#define _CLSPARSE_CSRMV_VECTOR_HPP_

#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/data_types/clvector.hpp"

template<typename T>
clsparseStatus
csrmv_vector(const clsparseScalarPrivate* pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const clsparseVectorPrivate* pX,
       const clsparseScalarPrivate* pBeta,
       clsparseVectorPrivate* pY,
       clsparseControl control)
{
    cl_uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);


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

/*
 * clsparse::array
 */
template<typename T>
clsparseStatus
csrmv_vector(const clsparse::vector<T>& pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const clsparse::vector<T>& pX,
       const clsparse::vector<T>& pBeta,
       clsparse::vector<T>& pY,
       clsparseControl control)
{
    cl_uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);


    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_general",
                                         "csrmv_general",
                                         params);
    KernelWrap kWrapper(kernel);

    cl_ulong offset  = 0;

    kWrapper << pMatx->m
             << pAlpha.data() << offset
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX.data() << offset
             << pBeta.data() << offset
             << pY.data() << offset;

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

#endif //_CLSPARSE_CSRMV_VECTOR_HPP_
