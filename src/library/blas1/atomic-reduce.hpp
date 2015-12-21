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
#ifndef _CLSPARSE_ATOMIC_REDUCE_HPP_
#define _CLSPARSE_ATOMIC_REDUCE_HPP_

#include <typeinfo>

#include "include/clSPARSE-private.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "reduce-operators.hpp"
#include "internal/data-types/clarray-base.hpp"

/* Helper function used in reduce type operations
 * pR = \sum pX
 * ASSUMPTIONS:
 *      pR initial value is set
 *      pX size is equal to wg_size;
 *      wg_size is the workgroup size
*/
template<typename T, ReduceOperator OP = RO_DUMMY>
clsparseStatus
atomic_reduce(clsparseScalarPrivate* pR,
              const cldenseVectorPrivate* pX,
              const clsparseIdx_t wg_size,
              const clsparseControl control)
{
    assert(wg_size == pX->num_values);
        
    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ReduceOperatorTrait<OP>::operation;

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

    if (typeid(cl_float) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_FLOAT";
        params.append(options);
    }
    else if (typeid(cl_double) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_DOUBLE";
        params.append(options);
    }
    else if (typeid(cl_int) == typeid(T) || typeid(clsparseIdx_t) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_INT";
        params.append(options);
    }
    else
    {
        return clsparseInvalidType;
    }

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "atomic_reduce", "reduce_block",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pR->value;
    kWrapper << pX->values;

    clsparseIdx_t blocksNum = (pX->num_values + wg_size - 1) / wg_size;
    clsparseIdx_t globalSize = blocksNum * wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(globalSize);

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
template<typename T, ReduceOperator OP = RO_DUMMY>
clsparseStatus
atomic_reduce(clsparse::array_base<T>& pR,
              const clsparse::array_base<T>& pX,
              const clsparseIdx_t wg_size,
              const clsparseControl control)
{
    assert(wg_size == pX.size());

    std::string params = std::string()
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ReduceOperatorTrait<OP>::operation;

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
    
    if (typeid(cl_float) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_FLOAT";
        params.append(options);
    }
    else if (typeid(cl_double) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_DOUBLE";
        params.append(options);
    }
    else if (typeid(cl_int) == typeid(T) || typeid(clsparseIdx_t) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_INT";
        params.append(options);
    }
    else
    {
        return clsparseInvalidType;
    }

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "atomic_reduce", "reduce_block",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pR.data();
    kWrapper << pX.data();

    clsparseIdx_t blocksNum = (pX.size() + wg_size - 1) / wg_size;
    clsparseIdx_t globalSize = blocksNum * wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


#endif //_CLSPARSE_ATOMIC_REDUCE_HPP_
