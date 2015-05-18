#ifndef _CLSPARSE_CONTROL_H_
#define _CLSPARSE_CONTROL_H_

#include "clSPARSE.h"
#include "../clsparseTimer/clsparseTimer.device.hpp"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif


struct _clsparseControl
{
    _clsparseControl( )
    { }

    _clsparseControl( const cl_command_queue& pQueue )
        : queue( pQueue ), event_wait_list( 0 )
    {
        // Initializing a cl::CommandQueue from a cl_command_queue does not appear to bump the refcount
        // Increment reference count since the library is caching a copy of the queue
        ::clRetainCommandQueue( pQueue );
    }

    // Destructor for queue should call release on it's own
    cl::CommandQueue queue;

    std::vector<cl::Event> event_wait_list;

    //it is better in that way;
    cl::Event event;

    // for NV(32) for AMD(64)
    size_t wavefront_size;
    size_t max_wg_size;

    // current device max compute units;
    cl_uint max_compute_units;

    //clSPARSE async execution; if true user is responsible to call for WaitForEvent;
    //otherwise after every kernel call we are syncing internally;
    cl_bool async;

    // Handle/pointer to the librar logger
    clsparseDeviceTimer* pDeviceTimer;

    cl::Context getContext( )
    {
        return queue.getInfo<CL_QUEUE_CONTEXT>( );
    }

};

#endif //_CLSPARSE_CONTROL_H_
