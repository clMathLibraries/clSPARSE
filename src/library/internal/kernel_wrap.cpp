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
    cl::Event tmp;
    cl_int status;
    try {

        status = queue.enqueueNDRangeKernel(kernel,
                                            cl::NullRange,
                                            global, local,
                                            &eventWaitList, &tmp);

        if (control->event != NULL)
        {
            auto refC = tmp.getInfo<CL_EVENT_REFERENCE_COUNT>();
            control->event = &tmp();
        }
        else
        {
            tmp.wait();
        }

    } catch (cl::Error e)
    {
        std::cout << "Kernel error: " << e.what() << std::endl;
        status = e.err();
    }


    return status;
}
