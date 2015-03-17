#include "clSPARSE.h"
//#include "internal/clsparse_internal.h"
#include "internal/clsparse_sources.hpp"
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

    cl_int status;
    clsparseStatus clsp_status;

    //validate input buffers
    clsp_status = validateMemObject(buff, sizeof(cl_float)*size);
    if(clsp_status != clsparseSuccess)
        return clsparseInvalidMemObj;
    clsp_status = validateMemObject(alpha, sizeof(cl_float));
    if(clsp_status != clsparseSuccess)
        return clsparseInvalidMemObj;


    //check opencl elements
    if (control->queue == NULL)
    {
        return clsparseInvalidCommandQueue;
    }

    //check event lists
    if ( (control->num_events_in_wait_list != 0)
         && (control->event_wait_list == NULL) )
    {
        return clsparseInvalidEventWaitList;
    }

    //context is already in control structure

    if (control->context == NULL)
    {
        printf("Context in control structure is null.\n");
        return clsparseInvalidContext;
    }


    static const std::string params =
            "-DINDEX_TYPE=int -DVALUE_TYPE=float -DSIZE_TYPE=int -DWG_SIZE=256";

    cl_kernel kernel = KernelCache::get(control->queue, "scale", params);

#ifndef NDEBUG
    std::cout << "params: " << params << std::endl;
#endif

    if (kernel == nullptr)
    {
        //free(key);
        return clsparseBuildProgramFailure;
    }


    KernelWrap kWrapper(kernel);

    kWrapper << buff
             << alpha
             << size;

    constexpr int BLOCK_SIZE = 256;

    int blocksNum = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int globalSize = blocksNum * BLOCK_SIZE;

    NDRange local(BLOCK_SIZE);
    NDRange global(globalSize);


    status = kWrapper.run(control->queue, global, local, nullptr, control->event);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}
