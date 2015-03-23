#include "kernel_wrap.hpp"

KernelWrap::KernelWrap(cl::Kernel &kernel) : kernel(kernel), argCounter(0)
{
}

cl_int KernelWrap::run(clsparseControl control,
                       const cl::NDRange global,
                       const cl::NDRange local)
{
    assert(argCounter == kernel.getInfo<CL_KERNEL_NUM_ARGS>());

    assert (local.dimensions() > 0);
    assert (global.dimensions() > 0);
    assert (local[0] > 0);
    assert (global[0] > 0);
    assert (global[0] >= local[0]);



    auto& queue = control->queue;
    const auto& eventWaitList = control->event_wait_list;
    cl_int status;
    try {
        cl::Event tmp;
        status = queue.enqueueNDRangeKernel( kernel,
                                                cl::NullRange,
                                                global, local,
                                                &eventWaitList, &tmp );

        if( control->event != NULL )
        {
            // Remember the event in our control structure
            *control->event = tmp( );

            // Prevent the reference count from hitting zero when the temporary goes out of scope
            ::clRetainEvent( *control->event );
        }
        else
        {
            tmp.wait( );
        }

    } catch (cl::Error e)
    {
        std::cout << "Kernel error: " << e.what() << std::endl;
        status = e.err();
    }


    return status;
}
