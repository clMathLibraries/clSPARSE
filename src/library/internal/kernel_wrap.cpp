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

//    auto ctx = queue.getInfo<CL_QUEUE_CONTEXT>();

//    cl::UserEvent myEvent(ctx, &status);
//    if(status != CL_SUCCESS)
//    {
//        std::cout << "myEvent create failed" << std::endl;
//    }

//    myEvent.setStatus(CL_RUNNING);
//    std::vector<cl::Event> myWlist(1);
//    myWlist[0] = myEvent;

    auto& queue = control->queue;
    auto& eventWaitList = control->event_wait_list;
    auto event = control->event;

    cl_int status;
    try {

//        status = queue.enqueueNDRangeKernel(kernel,
//                                            cl::NullRange,
//                                            global, local,
//                                            &events, &event);

    } catch (cl::Error& e)
    {
        std::cout << "Kernel error: " << e.what() << std::endl;
        status = e.err();
    }
    return status;
}
