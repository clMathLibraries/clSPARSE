#include "kernel_wrap.hpp"
#include "clsparse.error.hpp"

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
    OPENCL_V_THROW( status, "queue.enqueueNDRangeKernel" );

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
        OPENCL_V_THROW( status, "tmp.wait" );
    }

    return status;
}
