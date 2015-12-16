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
#ifndef _CLSPARSE_PRECOND_UTILS_HPP_
#define _CLSPARSE_PRECOND_UTILS_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include "internal/data-types/clvector.hpp"

template<typename T, bool inverse = false>
clsparseStatus
extract_diagonal(cldenseVectorPrivate* pDiag,
                 const clsparseCsrMatrixPrivate* pA,
                 clsparseControl control)
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

    assert (pA->num_rows > 0);
    assert (pA->num_cols > 0);
    assert (pA->num_nonzeros > 0);

    assert (pDiag->num_values == std::min(pA->num_rows, pA->num_cols));

    clsparseIdx_t wg_size = 256;
    clsparseIdx_t size = pA->num_rows;

    clsparseIdx_t nnz_per_row = pA->nnz_per_row();
    clsparseIdx_t wave_size = control->wavefront_size;
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


    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
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

    if (inverse)
        params.append(" -DOP_DIAG_INVERSE");

    if(typeid(T) == typeid(cl_double))
    {
        params.append(" -DDOUBLE");
        if (!control->dpfp_support)
        {
#ifndef NDEBUG
            std::cerr << "Failure attempting to run double precision kernel on device without DPFP support." << std::endl;
#endif
            return clsparseInvalidDevice;
        }
    }

    cl::Kernel kernel = KernelCache::get(control->queue, "matrix_utils",
                                         "extract_diagonal", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pDiag->values
             << pA->rowOffsets
             << pA->colIndices
             << pA->values;

    clsparseIdx_t predicted = subwave_size * size;

    clsparseIdx_t global_work_size =
            wg_size * ((predicted + wg_size - 1 ) / wg_size);
    cl::NDRange local(wg_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


template<typename T, bool inverse = false>
clsparseStatus
extract_diagonal(clsparse::vector<T>& pDiag,
                 const clsparseCsrMatrixPrivate* pA,
                 clsparseControl control)
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

    assert( pA->num_rows > 0 );
    assert( pA->num_cols > 0 );
    assert( pA->num_nonzeros > 0 );

    assert( pDiag.size( ) == std::min( pA->num_cols, pA->num_rows ) );

    clsparseIdx_t wg_size = 256;
    clsparseIdx_t size = pA->num_rows;

    clsparseIdx_t nnz_per_row = pA->nnz_per_row();
    clsparseIdx_t wave_size = control->wavefront_size;
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


    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
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

    if (inverse)
        params.append(" -DOP_DIAG_INVERSE");

    if(typeid(T) == typeid(cl_double))
    {
        params.append(" -DDOUBLE");
        if (!control->dpfp_support)
        {
#ifndef NDEBUG
            std::cerr << "Failure attempting to run double precision kernel on device without DPFP support." << std::endl;
#endif
            return clsparseInvalidDevice;
        }
    }

    cl::Kernel kernel = KernelCache::get(control->queue, "matrix_utils",
                                         "extract_diagonal", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pDiag.data()
             << pA->rowOffsets
             << pA->colIndices
             << pA->values;

    clsparseIdx_t predicted = subwave_size * size;

    clsparseIdx_t global_work_size =
            wg_size * ((predicted + wg_size - 1 ) / wg_size);
    cl::NDRange local(wg_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}
#endif //_CLSPARSE_PRECOND_UTILS_HPP_
