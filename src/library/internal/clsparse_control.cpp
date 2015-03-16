#include "clSPARSE.h"
#include "clsparse_control.hpp"
#include <malloc.h>

clsparseControl
clsparseCreateControl(cl_command_queue queue, cl_int *status)
{
    clsparseControl control = (clsparseControl)malloc(sizeof(_clsparseControl));

    cl_int err;
    if (!control)
    {
        control = NULL;
        err = clsparseOutOfHostMemory;
    }

    control->queue = queue;

    err = clGetCommandQueueInfo(control->queue, CL_QUEUE_CONTEXT,
                          sizeof(control->context), &(control->context), NULL);

    if (status != NULL)
    {
        *status = err;
    }

    control->event = NULL;
    control->num_events_in_wait_list = 0;
    control->event_wait_list = NULL;

    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

    return control;
}


clsparseStatus
clsparseReleaseControl(clsparseControl control)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->context = NULL;
    control->queue = NULL;
    control->num_events_in_wait_list = 0;
    control->event_wait_list = NULL;
    control->event = NULL;

    control->off_alpha = 0;
    control->off_beta = 0;
    control->off_x = 0;
    control->off_y = 0;

    free(control);

    control = NULL;

    return clsparseSuccess;
}

clsparseStatus
clsparseEventsToSync(clsparseControl control, cl_uint num_events_in_wait_list, cl_event *event_wait_list, cl_event *event)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    control->num_events_in_wait_list = num_events_in_wait_list;
    control->event_wait_list = event_wait_list;
    control->event = event;

    return clsparseSuccess;
}

clsparseStatus
clsparseSynchronize(clsparseControl control)
{
    if(control == NULL)
    {
        return clsparseInvalidControlObject;
    }

    cl_int sync_status = clWaitForEvents(1, control->event);
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
