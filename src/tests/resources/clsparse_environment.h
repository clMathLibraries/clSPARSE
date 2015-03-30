#ifndef _CL_SPARSE_ENVIRONMENT_H_
#define _CL_SPARSE_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>
#include "opencl_utils.h"


class ClSparseEnvironment : public ::testing::Environment
{
public:

    ClSparseEnvironment(cl_platform_type pID = AMD, cl_uint dID = 0)
    {
        // init cl environment
        cl_int status = CL_SUCCESS;

        cl::Device device = getDevice(pID, dID);

        std::cout << "Using device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        context = clCreateContext(NULL, 1, &device(), NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device(), 0, NULL);

        clsparseSetup();

        control = clsparseCreateControl(queue, NULL);

        //free(platforms);
    }


    void SetUp()
    {
    }

    //cleanup
    void TearDown()
    {
        //release cl structures

        clReleaseCommandQueue(queue);
        clReleaseContext(context);
        cl_int status  = clsparseReleaseControl(control);
        if (status != CL_SUCCESS)
        {
            std::cout << "Problem with releasing control object" << std::endl;
        }

        clsparseTeardown();
    }

    static cl_context context;
    static cl_command_queue queue;
    static clsparseControl control;

};


#endif //_CL_SPARSE_ENVIRONMENT_H_
