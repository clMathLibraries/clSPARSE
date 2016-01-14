/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_HPP_
#define _CLSPARSE_CSRMV_VECTOR_HPP_

#include "internal/clsparse-validate.hpp"
#include "internal/clsparse-internal.hpp"
#include "internal/data-types/clvector.hpp"

template<typename T>
clsparseStatus
csrmv_vector(const clsparseScalarPrivate* pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const cldenseVectorPrivate* pX,
       const clsparseScalarPrivate* pBeta,
       cldenseVectorPrivate* pY,
       clsparseControl control)
{
    clsparseIdx_t nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    clsparseIdx_t wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    clsparseIdx_t subwave_size = wave_size;

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

    std::string params = std::string() +
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<clsparseIdx_t>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    if (sizeof(clsparseIdx_t) == 8)
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_ulong>::type;
        params.append(options);
    }
    else
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_uint>::type;
        params.append(options);
    }

    if(typeid(T) == typeid(cl_double))
    {
        params += " -DDOUBLE";
        if (!control->dpfp_support)
        {
#ifndef NDEBUG
            std::cerr << "Failure attempting to run double precision kernel on device without DPFP support." << std::endl;
#endif
            return clsparseInvalidDevice;
        }
    }
    if(control->extended_precision)
        params += " -DEXTENDED_PRECISION";

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_general",
                                         "csrmv_general",
                                         params);
    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->num_rows
             << pAlpha->value << pAlpha->offset()
             << pMatx->row_pointer
             << pMatx->col_indices
             << pMatx->values
             << pX->values << pX->offset()
             << pBeta->value << pBeta->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    clsparseIdx_t predicted = subwave_size * pMatx->num_rows;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    clsparseIdx_t global_work_size =
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
csrmv_vector(const clsparse::array_base<T>& pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const clsparse::array_base<T>& pX,
       const clsparse::array_base<T>& pBeta,
       clsparse::array_base<T>& pY,
       clsparseControl control)
{
    clsparseIdx_t nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    clsparseIdx_t wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    clsparseIdx_t subwave_size = wave_size;

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

    std::string params = std::string() +
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    if (sizeof(clsparseIdx_t) == 8)
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type;
        params.append(options);
    }
    else
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_uint>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_uint>::type;
        params.append(options);
    }

    if(typeid(T) == typeid(cl_double))
    {
        params += " -DDOUBLE";
        if (!control->dpfp_support)
        {
#ifndef NDEBUG
            std::cerr << "Failure attempting to run double precision kernel on device without DPFP support." << std::endl;
#endif
            return clsparseInvalidDevice;
        }
    }
    if(control->extended_precision)
        params += " -DEXTENDED_PRECISION";

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_general",
                                         "csrmv_general",
                                         params);
    KernelWrap kWrapper(kernel);

    clsparseIdx_t offset = 0;

    kWrapper << pMatx->num_rows
             << pAlpha.data() << offset
             << pMatx->row_pointer
             << pMatx->col_indices
             << pMatx->values
             << pX.data() << offset
             << pBeta.data() << offset
             << pY.data() << offset;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    clsparseIdx_t predicted = subwave_size * pMatx->num_rows;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    clsparseIdx_t global_work_size =
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
