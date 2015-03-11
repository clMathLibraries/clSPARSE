#ifndef _CLSPARSE_CONTROL_H_
#define _CLSPARSE_CONTROL_H_

#include "clSPARSE.h"


typedef struct _clsparseControl
{
    cl_command_queue queue;
    cl_context context;

    cl_uint num_events_in_wait_list;
    cl_event *event_wait_list;
    cl_event *event;

} _clsparseControl;

#endif //_CLSPARSE_CONTROL_H_
