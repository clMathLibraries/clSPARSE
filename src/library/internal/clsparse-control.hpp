/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2015 Vratis, Ltd.
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#ifndef _CLSPARSE_CONTROL_H_
#define _CLSPARSE_CONTROL_H_

#include "clSPARSE.h"
#include "../clsparseTimer/clsparseTimer-device.hpp"

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
        : queue( pQueue ), event_wait_list( 0 ), pDeviceTimer( nullptr ), wavefront_size( 0 ),
        max_wg_size( 0 ), max_compute_units( 0 ), async( CL_FALSE )
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

    // Should we attempt to perform compensated summation?
    cl_bool extended_precision;

    // Does our device have double precision support?
    cl_bool dpfp_support;

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
