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

    //operation parameters
    size_t off_alpha;
    size_t off_beta;
    size_t off_x;
    size_t off_y;

} _clsparseControl;

#endif //_CLSPARSE_CONTROL_H_
