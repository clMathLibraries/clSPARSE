#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"


clsparseStatus
radix_sort_by_key(int keys_first, 
                  int keys_last,
                  int values_first,
                  cl_mem clInputKeys,
                  cl_mem clInputValues,
                  cl_mem clInputValues2,
                  int float_type, 
                  clsparseControl control)
{


    const int RADIX = 4; //Now you cannot replace this with Radix 8 since there is a
                         //local array of 16 elements in the histogram kernel.

    const int RADICES = (1 << RADIX); //Values handeled by each work-item?

    //int orig_szElements = static_cast<int>(std::distance(keys_first, keys_last));
    int orig_szElements = keys_last - keys_first + 1; 
    int szElements = orig_szElements;

    //place holder

    int kernel0_WgSize = 256;
    int kernel1_WgSize = 256;
    int kernel2_WgSize = 256;
    std::string params; 

    if(float_type == 0){ //TODO make it a template with type
      params = std::string() +
              " -DKERNEL0WORKGROUPSIZE=" + std::to_string(kernel0_WgSize)
            + " -DKERNEL1WORKGROUPSIZE=" + std::to_string(kernel1_WgSize)
            + " -DKERNEL2WORKGROUPSIZE=" + std::to_string(kernel2_WgSize)
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    } else {
      params = std::string() +
              " -DKERNEL0WORKGROUPSIZE=" + std::to_string(kernel0_WgSize)
            + " -DKERNEL1WORKGROUPSIZE=" + std::to_string(kernel1_WgSize)
            + " -DKERNEL2WORKGROUPSIZE=" + std::to_string(kernel2_WgSize)
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type;
    }

    //int computeUnits     = ctl.getDevice().getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    cl::Context context = control->getContext();
    std::vector<cl::Device> dev = context.getInfo<CL_CONTEXT_DEVICES>();
    int computeUnits = dev[0].getInfo< CL_DEVICE_MAX_COMPUTE_UNITS >( );

    if (computeUnits > 32 )
        computeUnits = 32;
    cl_int l_Error = CL_SUCCESS;

    int localSize  = 256;
    int wavefronts = 8;
    int numGroups = computeUnits * wavefronts;

    cl_mem clSwapKeys    = clCreateBuffer(context(),CL_MEM_READ_WRITE, (szElements)*sizeof(int), NULL, NULL );
    cl_mem clSwapValues  = clCreateBuffer(context(),CL_MEM_READ_WRITE, (szElements)*sizeof(int), NULL, NULL );
    cl_mem clSwapValues2;
    if(float_type == 0)
       clSwapValues2 = clCreateBuffer(context(),CL_MEM_READ_WRITE, (szElements)*sizeof(cl_float), NULL, NULL );
    else 
       clSwapValues2 = clCreateBuffer(context(),CL_MEM_READ_WRITE, (szElements)*sizeof(cl_double), NULL, NULL );

    cl_mem clHistData    = clCreateBuffer(context(),CL_MEM_READ_WRITE, (localSize * RADICES)*sizeof(unsigned int), NULL, NULL );

    ::cl::Kernel histKernel;
    ::cl::Kernel histSignedKernel;
    ::cl::Kernel permuteKernel;
    ::cl::Kernel permuteSignedKernel;
    ::cl::Kernel scanLocalKernel;


    int swap = 0;
    const int ELEMENTS_PER_WORK_ITEM = 4;
    int blockSize = (int)(ELEMENTS_PER_WORK_ITEM*localSize);//set at 1024
    int nBlocks = (int)(szElements + blockSize-1)/(blockSize);
	
    struct b3ConstData
    {
      int m_n;
      int m_nWGs;
      int m_startBit;
      int m_nBlocksPerWG;
    };
    b3ConstData cdata;

    cdata.m_n = (int)szElements;
    cdata.m_nWGs = (int)numGroups;
    //cdata.m_startBit = shift; //Shift value is set inside the for loop.
    cdata.m_nBlocksPerWG = (int)(nBlocks + numGroups - 1)/numGroups;
	
    if(nBlocks < numGroups)
    {
      cdata.m_nBlocksPerWG = 1;
      numGroups = nBlocks;
      cdata.m_nWGs = numGroups;
    }

    //histKernel
    cl::Kernel kernel0 = KernelCache::get(control->queue,"sort_by_key_common", "histogramAscInstantiated", params);
    //histSignedKernel
    cl::Kernel kernel1 = KernelCache::get(control->queue,"sort_by_key_common", "histogramSignedAscInstantiated", params);
    //scanLocalKernel
    cl::Kernel kernel2 = KernelCache::get(control->queue,"sort_by_key_common", "scanInstantiated", params);
    //permuteKernel
    cl::Kernel kernel3 = KernelCache::get(control->queue,"sort_by_key_uint", "permuteByKeyAscTemplate", params);
    //permuteSignedKernel

    cl::Kernel kernel4 = KernelCache::get(control->queue,"sort_by_key_int", "permuteByKeySignedAscTemplate", params);
  
    cl::NDRange local0(localSize);
    cl::NDRange global0(numGroups*localSize);
	
    cl::NDRange local1(localSize);
    cl::NDRange global1(numGroups*localSize);
	
    cl::NDRange local2(localSize);
    cl::NDRange global2(localSize);
	
    cl::NDRange local3(localSize);
    cl::NDRange global3(numGroups*localSize);
	
    cl::NDRange local4(localSize);
    cl::NDRange global4(numGroups*localSize);

    cl_int status;
    int bits = 0;
    for(bits = 0; bits < (sizeof(int) * 7)/*Bits per Byte*/; bits += RADIX)
    {
        KernelWrap kWrapper0(kernel0);
        KernelWrap kWrapper2(kernel2);
        KernelWrap kWrapper3(kernel3);   
       //Launch Kernel
        cdata.m_startBit = bits;
        //Histogram Kernel
        if(swap == 0)
           kWrapper0 << clInputKeys << clHistData << cdata.m_n << cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
        else
           kWrapper0 << clSwapKeys << clHistData << cdata.m_n <<  cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
 		
        status = kWrapper0.run(control, global0, local0);
       
        if (status != CL_SUCCESS)
        {
          return clsparseInvalidKernelExecution;
        }
		
        //Launch Local Scan Kernel
        kWrapper2 << clHistData << (int)numGroups;
		
        status = kWrapper2.run(control, global2, local2);
        		
        if (status != CL_SUCCESS)
        {
          return clsparseInvalidKernelExecution;
        }
		

        //Launch Permute Kernel
        if (swap == 0)
           kWrapper3 << clInputKeys <<  clInputValues << clInputValues2 << clHistData << clSwapKeys << clSwapValues << clSwapValues2 <<cdata.m_n << cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
        else
           kWrapper3 << clSwapKeys <<  clSwapValues << clSwapValues2 << clHistData << clInputKeys << clInputValues  << clInputValues2 << cdata.m_n << cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
	   
        status = kWrapper3.run(control, global3, local3);
		
        if (status != CL_SUCCESS)
        {
           return clsparseInvalidKernelExecution;
        }
		
        /*For swapping the buffers*/
        swap = swap? 0: 1;
    }
    //Perform Signed nibble radix sort operations here operations here
    { 
        //Histogram Kernel
        KernelWrap kWrapper1(kernel1);
        KernelWrap kWrapper2(kernel2);
        KernelWrap kWrapper4(kernel4);
        cdata.m_startBit = bits;

        kWrapper1 << clSwapKeys << clHistData << cdata.m_n << cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
		
        status = kWrapper1.run(control, global1, local1);

        if (status != CL_SUCCESS)
        {
			return clsparseInvalidKernelExecution;
        }
		
        kWrapper2 << clHistData << (int)numGroups;
        //Launch Scan Kernel
        status = kWrapper2.run(control, global2, local2);

	if (status != CL_SUCCESS)
        {
           return clsparseInvalidKernelExecution;
        }

        //Launch Permute Kernel
        kWrapper4 << clSwapKeys << clSwapValues << clSwapValues2 << clHistData << clInputKeys << clInputValues << clInputValues2 <<  cdata.m_n << cdata.m_nWGs << cdata.m_startBit << cdata.m_nBlocksPerWG;
        status = kWrapper4.run(control, global4, local4);
		
        if (status != CL_SUCCESS)
        {
           return clsparseInvalidKernelExecution;
        }
		
    }

    return clsparseSuccess;
}
