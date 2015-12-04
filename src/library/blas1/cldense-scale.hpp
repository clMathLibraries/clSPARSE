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
#ifndef _CLSPARSE_SCALE_HPP_
#define _CLSPARSE_SCALE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include "internal/data-types/clvector.hpp"

//TODO:: add offset to the scale kernel
template <typename T>
clsparseStatus
scale( clsparse::array_base<T>& pResult,
       const clsparse::array_base<T>& pAlpha,
       const clsparse::array_base<T>& pVector,
       clsparseControl control)
{
    const int group_size = 256;
    //const int group_size = control->max_wg_size;

    std::string params = std::string()
            + " -DVALUE_TYPE="+ OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(group_size);

    if (control->addressBits == GPUADDRESS64WORD)
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
                                         "blas1", "scale",
                                         params);
    KernelWrap kWrapper(kernel);

    cl_ulong size = pResult.size();
    cl_ulong offset = 0;

    kWrapper << size
             << pResult.data()
             << offset
             << pVector.data()
             << offset
             << pAlpha.data()
             << offset;

    clsparseIdx_t blocksNum = (size + group_size - 1) / group_size;
    clsparseIdx_t globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global (globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_SCALE_HPP_
