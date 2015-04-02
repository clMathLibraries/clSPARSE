#include "clSPARSE.h"
#include "clsparse_control.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include <iostream>
#include <malloc.h>
//get the wavefront size and max work group size
void collectEnvParams(clsparseControl control)
{
    if(!clsparseInitialized)
    {
        return;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return;
    }

    //Query device if necessary;
    cl::Device device = control->queue.getInfo<CL_QUEUE_DEVICE>();
    control->max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();

    const int wg_size = control->max_wg_size;

    int blocksNum = (1 + wg_size - 1) / wg_size;
    int globalSize = blocksNum * wg_size;

    const std::string params = std::string() +
            "-DWG_SIZE=" + std::to_string(wg_size);

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "control",
                                         "control",
                                         params);

    control->wavefront_size =
            kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(device);

}

clsparseControl
clsparseCreateControl(cl_command_queue& queue, cl_int *status)
{
    //clsparseControl control = (clsparseControl)malloc(sizeof(_clsparseControl));
    clsparseControl control = new _clsparseControl(queue);

    cl_int err;
    if (!control)
    {
        control = nullptr;
        err = clsparseOutOfHostMemory;
    }

    control->event = nullptr;
    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

    control->wavefront_size = 0;
    control->max_wg_size = 0;
    control->async = false;

    collectEnvParams(control);

    if (status != NULL)
    {
        *status = err;
    }

    return control;
}

clsparseStatus
clsparseEnableAsync(clsparseControl control, cl_bool async)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->async = async;
    return clsparseSuccess;
}

clsparseStatus
clsparseReleaseControl(clsparseControl control)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

//    if(control->event != nullptr)
//    {
//        delete control->event;
//    }

    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

    control->wavefront_size = 0;
    control->max_wg_size = 0;
    control->async = false;

    free(control);

    control = NULL;

    return clsparseSuccess;
}

clsparseStatus
clsparseSetupEventWaitList(clsparseControl control,
                           cl_uint num_events_in_wait_list,
                           cl_event *event_wait_list)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->event_wait_list.clear();
    control->event_wait_list.resize(num_events_in_wait_list);
    for (int i = 0; i < num_events_in_wait_list; i++)
    {
        control->event_wait_list[i] = event_wait_list[i];
    }
    control->event_wait_list.shrink_to_fit();

    return clsparseSuccess;
}

clsparseStatus
clsparseGetEvent(clsparseControl control, cl_event *event)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    *event = control->event( );
    return clsparseSuccess;

}

//clsparseStatus
//clsparseSetupEvent(clsparseControl control, cl_event* event)
//{
//    if(control == NULL)
//    {
//        return clsparseInvalidControlObject;
//    }

//    control->event = event;

//    return clsparseSuccess;
//}


//clsparseStatus
//clsparseSynchronize(clsparseControl control)
//{
//    if(control == NULL)
//    {
//        return clsparseInvalidControlObject;
//    }

//    cl_int sync_status = CL_SUCCESS;
//    try
//    {
//        // If the user registered an event with us
//        if( control->event )
//        {
//            // If the event is valid
//            if( *control->event )
//                ::clWaitForEvents( 1, control->event );
//        }
//    } catch (cl::Error e)
//    {
//        std::cout << "clsparseSynchronize error " << e.what() << std::endl;
//        sync_status = e.err();
//    }

//    if (sync_status != CL_SUCCESS)
//    {
//        return clsparseInvalidEvent;
//    }

//    return clsparseSuccess;
//}

clsparseStatus
clsparseSetOffsets(clsparseControl control,
                   size_t off_alpha, size_t off_beta,
                   size_t off_x, size_t off_y)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->off_alpha = off_alpha;
    control->off_beta = off_beta;
    control->off_x = off_x;
    control->off_y = off_y;

    return clsparseSuccess;
}
