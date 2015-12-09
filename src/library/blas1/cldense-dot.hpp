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
#ifndef _CLSPARSE_DOT_HPP_
#define _CLSPARSE_DOT_HPP_

#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include "commons.hpp"
#include "atomic-reduce.hpp"
#include "internal/data-types/clvector.hpp"

template<typename T>
clsparseStatus
inner_product (cldenseVectorPrivate* partial,
     const cldenseVectorPrivate* pX,
     const cldenseVectorPrivate* pY,
     const clsparseIdx_t size,
     const clsparseIdx_t REDUCE_BLOCKS_NUMBER,
     const clsparseIdx_t REDUCE_BLOCK_SIZE,
     const clsparseControl control)
{

    clsparseIdx_t nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DN_THREADS=" + std::to_string(nthreads);

    if (sizeof(clsparseIdx_t) == 8)
    {
        std::string options = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type;
        params.append(options);
    }
    else
    {
        std::string options = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_uint>::type;
        params.append(options);
    }

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

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "inner_product", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << partial->values
             << pX->values
             << pY->values;

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

template<typename T>
clsparseStatus dot(clsparseScalarPrivate* pR,
                   const cldenseVectorPrivate* pX,
                   const cldenseVectorPrivate* pY,
                   const clsparseControl control)
{

    cl_int status;

    init_scalar(pR, (T)0, control);

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const clsparseIdx_t REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */
    const clsparseIdx_t REDUCE_BLOCK_SIZE = 256;

    clsparseIdx_t xSize = pX->num_values - pX->offset();
    clsparseIdx_t ySize = pY->num_values - pY->offset();

    assert (xSize == ySize);

    clsparseIdx_t size = xSize;


    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        cldenseVectorPrivate partial;
        clsparseInitVector(&partial);
        partial.num_values = REDUCE_BLOCKS_NUMBER;

        clMemRAII<T> rPartial (control->queue(), &partial.values, partial.num_values);

        status = inner_product<T>(&partial, pX, pY, size,  REDUCE_BLOCKS_NUMBER,
                               REDUCE_BLOCK_SIZE, control);

        if (status != clsparseSuccess)
        {
            return clsparseInvalidKernelExecution;
        }

       status = atomic_reduce<T>(pR, &partial, REDUCE_BLOCK_SIZE,
                                     control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;
}

/*
 * clsparse::array
 */
template<typename T>
clsparseStatus
inner_product (clsparse::array_base<T>& partial,
     const clsparse::array_base<T>& pX,
     const clsparse::array_base<T>& pY,
     const clsparseIdx_t size,
     const clsparseIdx_t REDUCE_BLOCKS_NUMBER,
     const clsparseIdx_t REDUCE_BLOCK_SIZE,
     const clsparseControl control)
{

    clsparseIdx_t nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DN_THREADS=" + std::to_string(nthreads);

    if (sizeof(clsparseIdx_t) == 8)
    {
        std::string options = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type;
        params.append(options);
    }
    else
    {
        std::string options = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_uint>::type;
        params.append(options);
    }

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

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "inner_product", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << partial.data()
             << pX.data()
             << pY.data();

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

template<typename T>
clsparseStatus dot(clsparse::array_base<T>& pR,
                   const clsparse::array_base<T>& pX,
                   const clsparse::array_base<T>& pY,
                   const clsparseControl control)
{

    cl_int status;

    //not necessary to have it, but remember to init the pR with the proper value
    init_scalar(pR, (T)0, control);

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const clsparseIdx_t REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */
    const clsparseIdx_t REDUCE_BLOCK_SIZE = 256;

    clsparseIdx_t xSize = pX.size();
    clsparseIdx_t ySize = pY.size();

    assert (xSize == ySize);

    clsparseIdx_t size = xSize;

    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        clsparse::vector<T> partial(control, REDUCE_BLOCKS_NUMBER, 0,
                                   CL_MEM_READ_WRITE, false);

        status = inner_product<T>(partial, pX, pY, size,  REDUCE_BLOCKS_NUMBER,
                               REDUCE_BLOCK_SIZE, control);

        if (status != clsparseSuccess)
        {
            return clsparseInvalidKernelExecution;
        }

       status = atomic_reduce<T>(pR, partial, REDUCE_BLOCK_SIZE,
                                     control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_DOT_HPP_
