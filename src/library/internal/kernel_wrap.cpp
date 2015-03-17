#include "kernel_wrap.hpp"

KernelWrap::KernelWrap(cl_kernel &kernel) : kernel(&kernel), argCounter(0)
{


    cl_int status = clGetKernelInfo (*this->kernel,
                                  CL_KERNEL_NUM_ARGS,
                                  sizeof(int), &numArgs, NULL);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with obtaining kernel num args" << std::endl;
    }

    //std::cout << "\tKN = " << numArgs << std::endl;
}


cl_int KernelWrap::run(cl_command_queue& queue, const NDRange global,
                       const NDRange local,
                       const std::vector<cl_event>* events,
                       cl_event* event)
{

    cl_int currentArgNum;
    cl_int status = clGetKernelInfo (*kernel, CL_KERNEL_NUM_ARGS,
                                     sizeof(cl_int), &currentArgNum, NULL);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with obtaining kernel num args in kernel->run" << std::endl;
        return status;
    }

    assert(currentArgNum == argCounter);

    assert (local.dimensions() > 0);
    assert (global.dimensions() > 0);
    assert (local[0] > 0);
    assert (global[0] > 0);
    assert (global[0] >= local[0]);

    cl_event tmp;
    status = clEnqueueNDRangeKernel(queue, *kernel,
                                    global.dimensions(), NULL, global,
                                    local,
                                    (events != nullptr) ? events->size() : 0,
                                    (events != nullptr && events->size() > 0 ) ? (cl_event*)events->front() : nullptr,
                                    (event != nullptr) ? &tmp : nullptr);

    if (event != NULL && status == CL_SUCCESS)
    {
        *event = tmp;
    }

    return status;
}
