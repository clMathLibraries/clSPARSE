#include "clSPARSE.h"
#include "clsparse_control.hpp"

#include <iostream>
#include <malloc.h>

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

    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

    if (status != NULL)
    {
        *status = err;
    }

    return control;
}


clsparseStatus
clsparseReleaseControl(clsparseControl control)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

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
clsparseSetupEvent(clsparseControl control, cl_event *event)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->event = *event;

    return clsparseSuccess;
}


clsparseStatus
clsparseSynchronize(clsparseControl control)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    cl_int sync_status = CL_SUCCESS;
    try {
        control->event.wait();
    } catch (cl::Error& e)
    {
        std::cout << "clsparseSynchronize error " << e.what() << std::endl;
        sync_status = e.err();
    }

    if (sync_status != CL_SUCCESS)
    {
        return clsparseInvalidEvent;
    }

    return clsparseSuccess;
}

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
