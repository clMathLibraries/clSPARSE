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
#ifndef _CLSPARSE_ELEMENTWISE_HPP_
#define _CLSPARSE_ELEMENTWISE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include "elementwise-operators.hpp"

//#include "internal/data-types/clvector.hpp"
//forward declaration of clsparse::vector class for proper interface
namespace clsparse
{
template <typename T> class array_base;
}

/* Elementwise operation on two vectors
*/

template<typename T, ElementWiseOperator OP>
clsparseStatus
elementwise_transform(cldenseVectorPrivate* r,
                      const cldenseVectorPrivate* x,
                      const cldenseVectorPrivate* y,
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

    assert(x->num_values == y->num_values);
    assert(x->num_values== r->num_values);

    cl_ulong size = x->num_values;
    cl_uint wg_size = 256;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "elementwise_transform",
                                         "transform", params);

    KernelWrap kWrapper (kernel);

    kWrapper << size << r->values << x->values << y->values;

    int blocks = (size + wg_size - 1) / wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(blocks * wg_size);

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
template<typename T, ElementWiseOperator OP>
clsparseStatus
elementwise_transform(clsparse::array_base<T>& r,
                      const clsparse::array_base<T>& x,
                      const clsparse::array_base<T>& y,
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

    assert(x.size() == y.size());
    assert(x.size() == r.size());

    cl_ulong size = x.size();
    cl_uint wg_size = 256;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "elementwise_transform",
                                         "transform", params);

    KernelWrap kWrapper (kernel);

    kWrapper << size << r.data() << x.data() << y.data();

    int blocks = (size + wg_size - 1) / wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(blocks * wg_size);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_ELEMENTWISE_HPP_
