#include "clSPARSE.h"
#include "clsparse_control.h"


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

    return control;
}


clsparseStatus
clsparseReleaseControl(clsparseControl control)
{
    if(control == NULL)
    {
        return CL_INVALID_MEM_OBJECT;
    }

    control->context = NULL;
    control->queue = NULL;
    control->num_events_in_wait_list = 0;
    control->event_wait_list = NULL;
    control->event = NULL;

    free(control);

    control = NULL;

    return CL_SUCCESS;
}

clsparseStatus
clsparseEventsToSync(clsparseControl control, cl_uint num_events_in_wait_list, cl_event *event_wait_list, cl_event *event)
{
    control->num_events_in_wait_list = num_events_in_wait_list;
    control->event_wait_list = event_wait_list;
    control->event = event;
}

clsparseStatus
clsparseSynchronize(clsparseControl control)
{
    return clWaitForEvents(1, control->event);
}
