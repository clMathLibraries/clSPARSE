#ifndef _CL_SPARSE_ENVIRONMENT_H_
#define _CL_SPARSE_ENVIRONMENT_H_

#include <gtest/gtest.h>

#include <clSPARSE.h>
#include "opencl_utils.h"



class ClSparseEnvironment : public ::testing::Environment
{
public:

    ClSparseEnvironment()
    {
        // init cl environment
        cl_uint status = CL_SUCCESS;
        cl_platform_id* platforms = NULL;
        cl_uint num_platforms = 0;

        status = getPlatforms(&platforms, &num_platforms);
        if (status != CL_SUCCESS)
        {
            std::cerr << "Problem with setting up platforms" << std::endl;
            exit(-1);
        }

        printPlatforms(platforms, num_platforms);

        cl_device_id device = NULL;
        status = getDevice(platforms[0], &device, CL_DEVICE_TYPE_GPU);
        if (status != CL_SUCCESS)
        {
            std::cerr <<"Problem with initializing GPU device" << std::endl;
            exit(-2);
        }

        printDeviceInfo(device);

        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);

        clsparseSetup();
        free(platforms);
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
        clsparseTeardown();
    }

    static cl_context context;
    static cl_command_queue queue;

};


#endif //_CL_SPARSE_ENVIRONMENT_H_
