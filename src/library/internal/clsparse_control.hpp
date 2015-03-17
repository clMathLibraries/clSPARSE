#ifndef _CLSPARSE_CONTROL_H_
#define _CLSPARSE_CONTROL_H_

#include "clSPARSE.h"
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif


typedef struct _clsparseControl
{
    _clsparseControl()
    {

    }

    _clsparseControl(const cl_command_queue& queue)
        : queue(queue), event_wait_list(0)
    {

    }

    cl::CommandQueue queue;

    std::vector<cl::Event> event_wait_list;
    cl::Event event;

    //operation parameters
    size_t off_alpha;
    size_t off_beta;
    size_t off_x;
    size_t off_y;

    cl::Context getContext()
    {
        return queue.getInfo<CL_QUEUE_CONTEXT>();
    }

} _clsparseControl;

#endif //_CLSPARSE_CONTROL_H_
