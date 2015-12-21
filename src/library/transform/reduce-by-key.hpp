/* ************************************************************************
 * Copyright 2015 AMD, Ltd.
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
#ifndef _CLSPARSE_REDUCE_BY_KEY_HPP_
#define _CLSPARSE_REDUCE_BY_KEY_HPP_

#include "internal/data-types/clvector.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"
#include "transform/scan.hpp"

//Code adopted from Bolt


namespace internal {

template <typename KeyVector, typename ValueVector>
clsparseStatus
reduce_by_key( KeyVector& keys_output, ValueVector& values_output,
               const KeyVector& keys_input, const ValueVector& values_input,
               clsparseControl control)
{

    typedef typename KeyVector::value_type KeyType;
    typedef typename ValueVector::value_type ValueType;
    typedef typename ValueVector::size_type SizeType;


    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    SizeType size = keys_input.size();

    SizeType KERNELWAVES = 4;
    SizeType WAVESIZE = control->wavefront_size;
    SizeType kernel_WgSize = WAVESIZE * KERNELWAVES;

    SizeType size_input = size;
    SizeType modWgSize = (size_input & (kernel_WgSize - 1));
    if (modWgSize)
    {
        size_input &= ~modWgSize;
        size_input += kernel_WgSize;
    }

    SizeType numWorkGroups = size_input / kernel_WgSize;

    SizeType size_scan = numWorkGroups;

    modWgSize = (size_scan & (kernel_WgSize-1));

    if( modWgSize )
    {
        size_scan &= ~modWgSize;
        size_scan += kernel_WgSize;
    }


    //this vector stores the places where input index is changing;
    //TODO: change names to offsetIndexVector and offsetValueVector;
    KeyVector offsetArray (control, size, 0, CL_MEM_READ_WRITE, false);
    ValueVector offsetValArray (control, size, 0, CL_MEM_READ_WRITE, false);


    // just to change that faster;
    std::string strProgram = "reduce_by_key";
    //  offset calculation
    {
        std::string params = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<ValueType>::type
                + " -DKEY_TYPE=" + OclTypeTraits<KeyType>::type
                + " -DWG_SIZE=" + std::to_string(kernel_WgSize);

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

        cl::Kernel kernel = KernelCache::get(control->queue, strProgram,
                                             "offset_calculation", params);

        if( typeid(SizeType) == typeid(cl_double)  ||
            typeid(ValueType) == typeid(cl_double) ||
            typeid(KeyType) == typeid(cl_double))
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

        KernelWrap kWrapper (kernel);

        kWrapper << keys_input.data()
                 << offsetArray.data()
                 << size;

        cl::NDRange global(size);
        cl::NDRange local (kernel_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: offset_calculation");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        clsparseStatus clsp_status = inclusive_scan<EW_PLUS>(offsetArray, offsetArray, control);

        if (clsp_status != clsparseSuccess)
            return clsp_status;
    }


    KeyVector keySumArray(control, size_scan, 0, CL_MEM_READ_WRITE, false);
    ValueVector preSumArray(control, size_scan, 0, CL_MEM_READ_WRITE, false);
    ValueVector postSumArray(control, size_scan, 0, CL_MEM_READ_WRITE, false);

    SizeType ldsKeySize = kernel_WgSize * sizeof (KeyType);
    SizeType ldsValueSize = kernel_WgSize * sizeof (ValueType);

    // per block scan by key
    {

        std::string params = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<ValueType>::type
                + " -DKEY_TYPE=" + OclTypeTraits<KeyType>::type
                + " -DWG_SIZE=" + std::to_string(kernel_WgSize);

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

        if( typeid(SizeType) == typeid(cl_double)  ||
            typeid(ValueType) == typeid(cl_double) ||
            typeid(KeyType) == typeid(cl_double))
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

        cl::Kernel kernel = KernelCache::get(control->queue, strProgram,
                                             "per_block_scan_by_key", params);

        KernelWrap kWrapper (kernel);

        kWrapper << offsetArray.data()
                 << values_input.data()
                 << offsetValArray.data()
                 << size
                 << cl::Local(ldsKeySize)
                 << cl::Local(ldsValueSize)
                 << keySumArray.data()
                 << preSumArray.data();

        cl::NDRange global (size_input);
        cl::NDRange local  (kernel_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: per_block_scan_by_key");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    SizeType workPerThread = size_scan / kernel_WgSize;

    // intra block inclusive scan by key
    {
        std::string params = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<ValueType>::type
                + " -DKEY_TYPE=" + OclTypeTraits<KeyType>::type
                + " -DWG_SIZE=" + std::to_string(kernel_WgSize);

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


        if( typeid(SizeType) == typeid(cl_double)  ||
            typeid(ValueType) == typeid(cl_double) ||
            typeid(KeyType) == typeid(cl_double))
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

        cl::Kernel kernel = KernelCache::get(control->queue, strProgram,
                                             "intra_block_inclusive_scan_by_key",
                                             params);

        KernelWrap kWrapper (kernel);

        kWrapper << keySumArray.data()
                 << preSumArray.data()
                 << postSumArray.data()
                 << numWorkGroups
                 << cl::Local(ldsKeySize)
                 << cl::Local(ldsValueSize)
                 << workPerThread;

        cl::NDRange global (kernel_WgSize);
        cl::NDRange local  (kernel_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: intra_block_inclusive_scan_by_key");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    // per block addition by key
    {
        std::string params = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<ValueType>::type
                + " -DKEY_TYPE=" + OclTypeTraits<KeyType>::type
                + " -DWG_SIZE=" + std::to_string(kernel_WgSize);

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

        if( typeid(SizeType) == typeid(cl_double)  ||
            typeid(ValueType) == typeid(cl_double) ||
            typeid(KeyType) == typeid(cl_double))
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

        cl::Kernel kernel = KernelCache::get(control->queue, strProgram,
                                             "per_block_addition_by_key",
                                             params);

        KernelWrap kWrapper (kernel);

        kWrapper << keySumArray.data()
                 << postSumArray.data()
                 << offsetArray.data()
                 << offsetValArray.data()
                 << size;

        cl::NDRange global (size_input);
        cl::NDRange local  (kernel_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: per_block_addition_by_key");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    // key value mapping
    {
        std::string params = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<ValueType>::type
                + " -DKEY_TYPE=" + OclTypeTraits<KeyType>::type
                + " -DWG_SIZE=" + std::to_string(kernel_WgSize);

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

        if( typeid(SizeType) == typeid(cl_double)  ||
            typeid(ValueType) == typeid(cl_double) ||
            typeid(KeyType) == typeid(cl_double))
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

        cl::Kernel kernel = KernelCache::get(control->queue, strProgram,
                                             "key_value_mapping",
                                             params);

        KernelWrap kWrapper (kernel);

        kWrapper  << keys_input.data()
                  << keys_output.data()
                  << values_output.data()
                  << offsetArray.data()
                  << offsetValArray.data()
                  << size;

        cl::NDRange global (size_input);
        cl::NDRange local  (kernel_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: key_value_mapping");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

    }

    return clsparseSuccess;

}

}//namespace internal




#endif //_CLSPARSE_REDUCE_BY_KEY_HPP_
