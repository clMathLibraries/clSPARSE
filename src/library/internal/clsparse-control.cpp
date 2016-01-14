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

#include "clSPARSE.h"
#include "clsparse-control.hpp"
#include "internal/clsparse-internal.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "loadDynamicLibrary.hpp"
#include "clsparseTimer-extern.hpp"

#include <iostream>

//get the wavefront size and max work group size
clsparseStatus collectEnvParams(clsparseControl control)
{
    if(!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if( control == nullptr )
    {
        return clsparseInvalidControlObject;
    }

    cl_int status;
    //Query device if necessary;
    cl::Device device = control->queue.getInfo<CL_QUEUE_DEVICE>(&status);
    if(status != CL_SUCCESS)
    {
        switch (status) {
        case CL_INVALID_COMMAND_QUEUE:
            return clsparseInvalidCommandQueue;
            break;
        case CL_INVALID_VALUE:
            return clsparseInvalidValue;
            break;
        case CL_OUT_OF_HOST_MEMORY:
            return clsparseOutOfHostMemory;
        case CL_OUT_OF_RESOURCES:
            return clsparseOutOfResources;
        default:
            return clsparseInvalidDevice;
            break;
        }
    }
    control->max_wg_size = device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>( );

    const size_t wg_size = control->max_wg_size;

    size_t blocksNum = ( 1 + wg_size - 1 ) / wg_size;
    size_t globalSize = blocksNum * wg_size;

    const std::string params = std::string( ) +
        "-DWG_SIZE=" + std::to_string( wg_size );

    cl::Kernel kernel = KernelCache::get( control->queue,
        "control",
        "control",
        params );

    control->wavefront_size =
        kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>( device );

    control->max_compute_units =
            device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();

    control->addressBits = device.getInfo<CL_DEVICE_ADDRESS_BITS>();

#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    if ( device.getInfo<CL_DEVICE_EXTENSIONS>( ).find("cl_khr_fp64") != std::string::npos ||
         device.getInfo<CL_DEVICE_EXTENSIONS>( ).find("cl_amd_fp64") != std::string::npos )
    {
        if (device.getInfo<CL_DEVICE_DOUBLE_FP_CONFIG>( ))
            control->dpfp_support = true;
    }
#endif

    return clsparseSuccess;
}

clsparseCreateResult
clsparseCreateControl( cl_command_queue queue )
{
    clsparseCreateResult cPair;
    cPair.status = clsparseSuccess;
    cPair.control = new _clsparseControl( queue );

    if( !cPair.control )
    {
        cPair.control = nullptr;
        cPair.status = clsparseOutOfHostMemory;
        return cPair;
    }

    cPair.control->event = nullptr;
    cPair.control->wavefront_size = 0;
    cPair.control->max_wg_size = 0;
    cPair.control->async = false;
    cPair.control->extended_precision = false;
    cPair.control->dpfp_support = false;
	cPair.control->addressBits = 64; // default 64 bits

    collectEnvParams( cPair.control );

    //	Discover and load the timer module if present
    void* timerLibHandle = LoadSharedLibrary( "lib", "clsparseTimer", false );
    if( timerLibHandle )
    {
        //	Timer module discovered and loaded successfully
        //	Initialize function pointers to call into the shared module
        // PFCLSPARSETIMER pfclsparseTimer = static_cast<PFCLSPARSETIMER> ( LoadFunctionAddr( timerLibHandle, "clsparseGetTimer" ) );
        void* funcPtr = LoadFunctionAddr( timerLibHandle, "clsparseGetTimer" );
        PFCLSPARSETIMER pfclsparseTimer = *static_cast<PFCLSPARSETIMER*>( static_cast<void*>( &funcPtr ) );

        //	Create and initialize our timer class, if the external timer shared library loaded
        if( pfclsparseTimer )
        {
            cPair.control->pDeviceTimer = static_cast<clsparseDeviceTimer*> ( pfclsparseTimer( CLSPARSE_GPU ) );
        }
    }

    return cPair;
}

clsparseStatus
clsparseEnableAsync( clsparseControl control, cl_bool async )
{
    if( control == NULL )
    {
        return clsparseInvalidControlObject;
    }

    control->async = async;
    return clsparseSuccess;
}

clsparseStatus
clsparseEnableExtendedPrecision( clsparseControl control, cl_bool extended_precision )
{
    if( control == NULL )
    {
        return clsparseInvalidControlObject;
    }

    control->extended_precision = extended_precision;
    return clsparseSuccess;
}

clsparseStatus
clsparseReleaseControl( clsparseControl control )
{
    if( control == NULL )
    {
        return clsparseInvalidControlObject;
    }

    //    if(control->event != nullptr)
    //    {
    //        delete control->event;
    //    }

    //    control->off_alpha = 0;
    //    control->off_beta = 0;
    //    control->off_x = 0;
    //    control->off_y = 0;

    control->wavefront_size = 0;
    control->max_wg_size = 0;
    control->async = false;

    delete control;

    control = NULL;

    return clsparseSuccess;
}

clsparseStatus
clsparseSetupEventWaitList( clsparseControl control,
cl_uint num_events_in_wait_list,
cl_event *event_wait_list )
{
    if( control == NULL )
    {
        return clsparseInvalidControlObject;
    }

    control->event_wait_list.clear( );
    control->event_wait_list.resize( num_events_in_wait_list );
    for( cl_uint i = 0; i < num_events_in_wait_list; i++ )
    {
        control->event_wait_list[ i ] = event_wait_list[ i ];
    }
    control->event_wait_list.shrink_to_fit( );

    return clsparseSuccess;
}

clsparseEventResult
clsparseGetEvent( clsparseControl control )
{
    clsparseEventResult resPair;
    resPair.status = clsparseSuccess;
    resPair.event = nullptr;

    if( control == NULL )
    {
        resPair.status = clsparseInvalidControlObject;
        return resPair;
    }

    //keeps the event valid on the user side
    ::clRetainEvent( control->event( ) );

    resPair.event = control->event( );

    return resPair;
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

//clsparseStatus
//clsparseSetOffsets(clsparseControl control,
//                   size_t off_alpha, size_t off_beta,
//                   size_t off_x, size_t off_y)
//{
//    if(control == NULL)
//    {
//        return clsparseInvalidControlObject;
//    }

//    control->off_alpha = off_alpha;
//    control->off_beta = off_beta;
//    control->off_x = off_x;
//    control->off_y = off_y;

//    return clsparseSuccess;
//}
