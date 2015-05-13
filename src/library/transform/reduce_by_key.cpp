#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "transform/transform_kernels.h"

#define KERNEL02WAVES 4
#define KERNEL1WAVES 4
#define WAVESIZE 64


clsparseStatus
reduce_by_key(
    int keys_first,
    int keys_last,
    int values_first,
    cl_mem keys_input, 
    cl_mem values_input,
    cl_mem keys_output,
    cl_mem values_output,
    int *count,
    clsparseControl control
)
{

    cl_int l_Error;

    /**********************************************************************************
     * Compile Options
     *********************************************************************************/
    const int kernel0_WgSize = WAVESIZE*KERNEL02WAVES;
    const int kernel1_WgSize = WAVESIZE*KERNEL1WAVES;
    const int kernel2_WgSize = WAVESIZE*KERNEL02WAVES;
	
    const std::string params = std::string() +
              " -DKERNEL0WORKGROUPSIZE=" + std::to_string(kernel0_WgSize)
            + " -DKERNEL1WORKGROUPSIZE=" + std::to_string(kernel1_WgSize)
            + " -DKERNEL2WORKGROUPSIZE=" + std::to_string(kernel2_WgSize);
	
    //int computeUnits     = ctl.getDevice( ).getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
 
    cl::Context context = control->getContext();
    std::vector<cl::Device> dev = context.getInfo<CL_CONTEXT_DEVICES>(); 
    int computeUnits  = dev[0].getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );
    int wgPerComputeUnit = dev[0].getInfo< CL_DEVICE_MAX_WORK_GROUP_SIZE >( );

    //int wgPerComputeUnit =  ctl.getWGPerComputeUnit( );

    int resultCnt = computeUnits * wgPerComputeUnit;

    //  Ceiling function to bump the size of input to the next whole wavefront size
    //cl_uint numElements = static_cast< cl_uint >( std::distance( keys_first, keys_last ) );
    cl_uint numElements = keys_last - keys_first + 1;
	
    size_t sizeInputBuff = numElements;
    int modWgSize = (sizeInputBuff & (kernel0_WgSize-1));
    if( modWgSize )
    {
        sizeInputBuff &= ~modWgSize;
        sizeInputBuff += kernel0_WgSize;
    }
    cl_uint numWorkGroupsK0 = static_cast< cl_uint >( sizeInputBuff / kernel0_WgSize );

    //  Ceiling function to bump the size of the sum array to the next whole wavefront size
    size_t sizeScanBuff = numWorkGroupsK0;
    modWgSize = (sizeScanBuff & (kernel0_WgSize-1));
    if( modWgSize )
    {
        sizeScanBuff &= ~modWgSize;
        sizeScanBuff += kernel0_WgSize;
    }

    // Create buffer wrappers so we can access the host functors, for read or writing in the kernel
    //device_vector< int > tempArray( numElements, 0, CL_MEM_READ_WRITE, false, ctl);
    //::cl::Buffer tempArrayVec = tempArray.begin( ).base().getContainer().getBuffer();
    cl_mem tempArrayVec = clCreateBuffer(context(),CL_MEM_READ_WRITE, (numElements)*sizeof(int), NULL, NULL );

    /**********************************************************************************
     *  Kernel 0
     *********************************************************************************/
	 
    cl::Kernel kernel0 = KernelCache::get(control->queue,"reduce_by_key", "OffsetCalculation", params);
	
    KernelWrap kWrapper0(kernel0);

    kWrapper0 << keys_input << tempArrayVec
              << numElements;

    cl::NDRange local0(kernel0_WgSize);
    cl::NDRange global0(sizeInputBuff);

    cl_int status = kWrapper0.run(control, global0, local0);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    int init = 0;

    scan(0, 
	 numElements - 1,
         tempArrayVec,
         tempArrayVec,
         0,
         0,
         control
         );

    int pattern = 0;
    cl_mem keySumArray = clCreateBuffer(context(),CL_MEM_READ_WRITE, (sizeScanBuff)*sizeof(int), NULL, NULL );
    cl_mem preSumArray = clCreateBuffer(context(),CL_MEM_READ_WRITE, (sizeScanBuff)*sizeof(int), NULL, NULL );
    cl_mem postSumArray = clCreateBuffer(context(),CL_MEM_READ_WRITE,(sizeScanBuff)*sizeof(int), NULL, NULL );
    clEnqueueFillBuffer(control->queue(), keySumArray, &pattern, sizeof(int), 0,
                        (sizeScanBuff)*sizeof(int), NULL, NULL, NULL);
    clEnqueueFillBuffer(control->queue(), preSumArray, &pattern, sizeof(int), 0,
                        (sizeScanBuff)*sizeof(int), NULL, NULL, NULL);
    clEnqueueFillBuffer(control->queue(), postSumArray, &pattern, sizeof(int), 0,
                        (sizeScanBuff)*sizeof(int), NULL, NULL, NULL);

    
    /**********************************************************************************
     *  Kernel 1
     *********************************************************************************/
	
    cl::Kernel kernel1 = KernelCache::get(control->queue,"reduce_by_key", "perBlockScanByKey", params);

    KernelWrap kWrapper1(kernel1);

    //cl_uint workPerThread = static_cast< cl_uint >( no_workgrs % kernel1_WgSize?((no_workgrs/kernel1_WgSize)+1):(no_workgrs/kernel1_WgSize) );

    kWrapper1 << tempArrayVec
	      << values_input
              << numElements
	      << keySumArray
	      << preSumArray;

    cl::NDRange local1(kernel0_WgSize);
    cl::NDRange global1(sizeInputBuff);

    status = kWrapper1.run(control, global1, local1);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
 
    /**********************************************************************************
     *  Kernel 2
     *********************************************************************************/
    cl_uint workPerThread = static_cast< cl_uint >( sizeScanBuff / kernel1_WgSize );
	
    cl::Kernel kernel2 = KernelCache::get(control->queue,"reduce_by_key", "intraBlockInclusiveScanByKey", params);

    KernelWrap kWrapper2(kernel2);

    kWrapper2 << keySumArray << preSumArray
              << postSumArray << numWorkGroupsK0 << workPerThread;

    cl::NDRange local2(kernel1_WgSize);
    cl::NDRange global2(kernel1_WgSize);

    status = kWrapper2.run(control, global2, local2);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    /**********************************************************************************
     *  Kernel 3
     *********************************************************************************/
	 
    cl::Kernel kernel3 = KernelCache::get(control->queue,"reduce_by_key", "keyValueMapping", params);

    KernelWrap kWrapper3(kernel3);
	
    kWrapper3 << keys_input << keys_output
              << values_input << values_output << tempArrayVec
              << keySumArray << postSumArray << numElements;

    cl::NDRange local3(kernel0_WgSize);
    cl::NDRange global3(sizeInputBuff);

    status = kWrapper3.run(control, global3, local3);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    int *h_result = (int *) malloc (sizeof(int));
	
    clEnqueueReadBuffer(control->queue(), 
                        tempArrayVec, 
                        1, 
                       (numElements-1)*sizeof(int),
                        sizeof(int), 
                        h_result, 
                        0, 
                        0, 
                        0);	

    *count = *(h_result);
    //printf("h_result = %d\n", *count );

    return clsparseSuccess;
}   //end of reduce_by_key
