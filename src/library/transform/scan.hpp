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
#ifndef _CLSPARSE_SCAN_HPP_
#define _CLSPARSE_SCAN_HPP_

#include "internal/data-types/clvector.hpp"
#include "blas1/elementwise-operators.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

//Code adopted from Bolt
//Code will be very usefull for future AMG implementation

namespace internal
{

template <ElementWiseOperator OP, typename VectorType>
clsparseStatus
scan(VectorType& output, const VectorType& input,
     clsparseControl control, bool exclusive)
{
    typedef typename VectorType::size_type SizeType; //check for cl_ulong
    typedef typename VectorType::value_type T;

    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    assert (input.size() == output.size());

    SizeType num_elements = input.size();

    //std::cout << "num_elements = " << num_elements << std::endl;

    SizeType KERNEL02WAVES = 4;
    SizeType KERNEL1WAVES = 4;
    SizeType WAVESIZE = control->wavefront_size;

    SizeType kernel0_WgSize = WAVESIZE*KERNEL02WAVES;
    SizeType kernel1_WgSize = WAVESIZE*KERNEL1WAVES;
    SizeType kernel2_WgSize = WAVESIZE*KERNEL02WAVES;

    SizeType numElementsRUP = num_elements;
    SizeType modWgSize = (numElementsRUP & ((kernel0_WgSize*2)-1));

    if( modWgSize )
    {
        numElementsRUP &= ~modWgSize;
        numElementsRUP += (kernel0_WgSize*2);
    }

    //2 element per work item
    SizeType numWorkGroupsK0 = numElementsRUP / (kernel0_WgSize*2);

    SizeType sizeScanBuff = numWorkGroupsK0;

    modWgSize = (sizeScanBuff & ((kernel0_WgSize*2)-1));
    if( modWgSize )
    {
        sizeScanBuff &= ~modWgSize;
        sizeScanBuff += (kernel0_WgSize*2);
    }

    cl::Context ctx = control->getContext();

    clsparse::vector<T> preSumArray(control, sizeScanBuff,
                                    0, CL_MEM_READ_WRITE, false);
    clsparse::vector<T> preSumArray1(control, sizeScanBuff,
                                     0, CL_MEM_READ_WRITE, false);
    clsparse::vector<T> postSumArray(control, sizeScanBuff,
                                     0, CL_MEM_READ_WRITE, false);

    T operator_identity = 0;

    //std::cout << "operator_identity = " << operator_identity << std::endl;
    //scan in blocks
    {
        //local mem size
        std::size_t lds = kernel0_WgSize * 2 * sizeof(T);

        const std::string params = std::string()
                + " -DSIZE_TYPE="  + OclTypeTraits<SizeType>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
                + " -DWG_SIZE="    + std::to_string(kernel0_WgSize)
                + " -D" + ElementWiseOperatorTrait<OP>::operation;


        cl::Kernel kernel = KernelCache::get(control->queue, "scan",
                                             "per_block_inclusive_scan", params);

        KernelWrap kWrapper(kernel);


        kWrapper << input.data()
                 << operator_identity
                 << (SizeType)input.size()
                 << cl::Local(lds)
                 << preSumArray.data()
                 << preSumArray1.data()
                 << (int) exclusive;

        cl::NDRange global(numElementsRUP/2);
        cl::NDRange local (kernel0_WgSize);

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: per_block_inclusive_scan");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

    }


    {
        //local mem size
        std::size_t lds = kernel0_WgSize * sizeof(T);

        SizeType workPerThread = sizeScanBuff / kernel1_WgSize;

        const std::string params = std::string()
                + " -DSIZE_TYPE="  + OclTypeTraits<SizeType>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
                + " -DWG_SIZE="    + std::to_string(kernel1_WgSize)
                + " -D" + ElementWiseOperatorTrait<OP>::operation;

        cl::Kernel kernel = KernelCache::get(control->queue, "scan",
                                             "intra_block_inclusive_scan", params);

        KernelWrap kWrapper(kernel);

        kWrapper << postSumArray.data()
                 << preSumArray.data()
                 << operator_identity
                 << numWorkGroupsK0
                 << cl::Local(lds)
                 << workPerThread;

        cl::NDRange global ( kernel1_WgSize );
        cl::NDRange local  ( kernel1_WgSize );

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: intra_block_inclusive_scan");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    {
        std::size_t lds = kernel0_WgSize * sizeof(T); //local mem size

        const std::string params = std::string()
                + " -DSIZE_TYPE="  + OclTypeTraits<SizeType>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
                + " -DWG_SIZE="    + std::to_string(kernel1_WgSize)
                + " -D" + ElementWiseOperatorTrait<OP>::operation;


        cl::Kernel kernel = KernelCache::get(control->queue, "scan",
                                             "per_block_addition", params);

        KernelWrap kWrapper(kernel);

        kWrapper << output.data()
                 << input.data()
                 << postSumArray.data()
                 << preSumArray1.data()
                 << cl::Local(lds)
                 << num_elements
                 << (int)exclusive
                 << operator_identity;

        cl::NDRange global ( numElementsRUP );
        cl::NDRange local  ( kernel2_WgSize );

        cl_int status = kWrapper.run(control, global, local);

        CLSPARSE_V(status, "Error: per_block_addition");

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;

}

} //namespace internal


template <ElementWiseOperator OP, typename T>
clsparseStatus
exclusive_scan( clsparse::vector<T>& output,
                const clsparse::vector<T>& input,
                    clsparseControl control)
{
   return internal::scan<OP>(output, input, control, true);
}

template <ElementWiseOperator OP, typename T>
clsparseStatus
inclusive_scan( clsparse::vector<T>& output,
                const clsparse::vector<T>& input,
                clsparseControl control)
{
  return internal::scan<OP>(output, input, control, false);
}



#endif //_CLSPARSE_SCAN_HPP_
