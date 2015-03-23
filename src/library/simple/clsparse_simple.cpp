#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"


#include <iostream>
#include <cassert>

clsparseStatus
clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
              clsparseControl control)
{
    if(!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    cl_int status;
    clsparseStatus clsp_status;

    //validate input buffers
    clsp_status = validateMemObject(buff, sizeof(cl_float)*size);
    if(clsp_status != clsparseSuccess)
        return clsparseInvalidMemObj;
    clsp_status = validateMemObject(alpha, sizeof(cl_float));
    if(clsp_status != clsparseSuccess)
        return clsparseInvalidMemObj;




    const int wg_size = 256;

    int blocksNum = (size + wg_size - 1) / wg_size;
    int globalSize = blocksNum * wg_size;


    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE="+ OclTypeTraits<cl_float>::type
            + " -DWG_SIZE=" + std::to_string(wg_size);

    cl::Kernel kernel = KernelCache::get(control->queue, "scale", params);

#ifndef NDEBUG
    std::cout << "params: " << params << std::endl;
#endif

    KernelWrap kWrapper(kernel);

    kWrapper << buff
             << alpha
             << size;


    cl::NDRange local(wg_size);
    cl::NDRange global(globalSize);

    status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}
