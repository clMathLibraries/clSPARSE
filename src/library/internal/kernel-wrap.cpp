/* ************************************************************************
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

#include "kernel-wrap.hpp"
#include "clSPARSE-error.h"

KernelWrap::KernelWrap( cl::Kernel &kernel ): kernel( kernel ), argCounter( 0 ), addrBits( 0 )
{
}

cl_int KernelWrap::run( clsparseControl control,
    const cl::NDRange global,
    const cl::NDRange local )
{
    assert( argCounter == kernel.getInfo<CL_KERNEL_NUM_ARGS>( ) );

    assert( local.dimensions( ) > 0 );
    assert( global.dimensions( ) > 0 );
    assert( local[ 0 ] > 0 );
    assert( global[ 0 ] > 0 );
    assert( global[ 0 ] >= local[ 0 ] );

    auto& queue = control->queue;
    const auto& eventWaitList = control->event_wait_list;
    cl_int status;

    cl::Event tmp;
    status = queue.enqueueNDRangeKernel( kernel,
        cl::NullRange,
        global, local,
        &eventWaitList, &tmp );
    CLSPARSE_V( status, "queue.enqueueNDRangeKernel" );

    if( control->pDeviceTimer )
    {
        control->pDeviceTimer->AddSample( std::vector < cl::Event > { tmp } );
    }

    if( control->async )
    {
        // Remember the event in our control structure
        // operator= takes the ownership
        control->event = tmp;
        // Prevent the reference count from hitting zero when the temporary goes out of scope
        //::clRetainEvent( control->event( ) );
    }
    else
    {
        status = tmp.wait( );
        CLSPARSE_V( status, "tmp.wait" );
    }

    return status;
}
