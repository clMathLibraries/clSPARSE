/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#include "clSPARSE.h"
#include "internal/clsparse-internal.hpp"
#include "internal/clsparse-validate.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"

#define KERNEL02WAVES 4
#define KERNEL1WAVES 4
#define HSAWAVES 4
#define WAVESIZE 64


clsparseStatus
scan( int first,
      int last,
      cl_mem input_array,
      cl_mem output_result,
      int init_T,
      int exclusive,
      clsparseControl control)
{

	/**********************************************************************************
	 * Compile Options
	 *********************************************************************************/

	const int kernel0_WgSize = WAVESIZE*KERNEL02WAVES;
	const int kernel1_WgSize = WAVESIZE*KERNEL1WAVES;
	const int kernel2_WgSize = WAVESIZE*KERNEL02WAVES;

	const std::string params = std::string() +
              " -DKERNEL0WORKGROUPSIZE=" + std::to_string(kernel0_WgSize)
            + " -DKERNEL1WORKGROUPSIZE=" + std::to_string(kernel1_WgSize)
            + " -DKERNEL2WORKGROUPSIZE=" + std::to_string(kernel2_WgSize)
            + " -DEXCLUSIVE=" + std::to_string(exclusive);


	/**********************************************************************************
	 * Round Up Number of Elements
	 *********************************************************************************/
	cl_uint  numElements = last - first + 1;

	//cl_uint computeUnits = ctrl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
	cl::Context context = control->getContext();
        std::vector<cl::Device> dev = context.getInfo<CL_CONTEXT_DEVICES>();
        int computeUnits  = dev[0].getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );

	unsigned int wgComputeUnit = (computeUnits*64); //64 boosts up the performance
	unsigned int load_per_wg = numElements % wgComputeUnit?((numElements/wgComputeUnit)+1):(numElements/wgComputeUnit);

	size_t modWgSize = (load_per_wg & ((kernel0_WgSize)-1));
	if( modWgSize )
	{
	   load_per_wg &= ~modWgSize;
	   load_per_wg += (kernel0_WgSize);
	}

	unsigned int no_workgrs = numElements % load_per_wg?((numElements/load_per_wg)+1):(numElements/load_per_wg);

        cl_mem preSumArray = clCreateBuffer(context(),CL_MEM_READ_WRITE, (no_workgrs)*sizeof(int), NULL, NULL );

	/**********************************************************************************
	 *  Kernel 0
	 *********************************************************************************/
	cl::Kernel kernel0 = KernelCache::get(control->queue,"scan_kernels", "perBlockInclusiveScan", params);

	KernelWrap kWrapper0(kernel0);

	kWrapper0 << input_array << init_T
                  << numElements << preSumArray << load_per_wg;

        cl::NDRange local0(kernel0_WgSize);
        cl::NDRange global0(no_workgrs * kernel0_WgSize);

        cl_int status = kWrapper0.run(control, global0, local0);

        if (status != CL_SUCCESS)
        {
        return clsparseInvalidKernelExecution;
        }

	/**********************************************************************************
	 *  Kernel 1
	*********************************************************************************/

	cl::Kernel kernel1 = KernelCache::get(control->queue,"scan_kernels", "intraBlockInclusiveScan", params);

	KernelWrap kWrapper1(kernel1);

	cl_uint workPerThread = static_cast< cl_uint >( no_workgrs % kernel1_WgSize?((no_workgrs/kernel1_WgSize)+1):(no_workgrs/kernel1_WgSize) );

	kWrapper1 << preSumArray << init_T
                  << no_workgrs << workPerThread;

        cl::NDRange local1(kernel1_WgSize);
        cl::NDRange global1(kernel1_WgSize);

        status = kWrapper1.run(control, global1, local1);

        if (status != CL_SUCCESS)
        {
          return clsparseInvalidKernelExecution;
        }

	/**********************************************************************************
	 *  Kernel 2
	 *********************************************************************************/

	cl::Kernel kernel2 = KernelCache::get(control->queue,"scan_kernels", "perBlockAddition", params);

	KernelWrap kWrapper2(kernel2);

	kWrapper2 << output_result << input_array
                  << preSumArray << numElements << load_per_wg << init_T;

        cl::NDRange local2(kernel2_WgSize);
        cl::NDRange global2(no_workgrs * kernel2_WgSize);

        status = kWrapper2.run(control, global2, local2);

        clReleaseMemObject(preSumArray);

        if (status != CL_SUCCESS)
        {
          return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;

}   //end
