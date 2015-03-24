#ifndef _CL_SPARSE_ENVIRONMENT_H_
#define _CL_SPARSE_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <map>
#include <clSPARSE.h>
#include "opencl_utils.h"

typedef enum _platform {
    AMD = 0,
    NVIDIA
} cl_platform_type;


class ClSparseEnvironment : public ::testing::Environment
{
public:

    ClSparseEnvironment(cl_platform_type pID = AMD, cl_uint dID = 0)
    {
        // init cl environment
        cl_int status = CL_SUCCESS;

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        assert(platforms.size() > 0);

        std::map<std::string, int> pNames;
        //search for AMD or NVIDIA
        cl_int pIndex = -1;
        for (const auto& p : platforms)
        {
            //p.getInfo returns null terminated char* in
            // strange format "blabla\000" I don't know how to get rid of it :/
            std::string pName = p.getInfo<CL_PLATFORM_NAME>();
            std::string name = pName.substr(0, pName.size()-1);
            pNames.insert(std::make_pair(name, ++pIndex));
        }

        //get index of desired platform;
        std::string desired_platform_name;
        if (pID == AMD)
        {
            desired_platform_name = amd_platform_str;
        }
        else if (pID == NVIDIA)
        {
            desired_platform_name = nvidia_platform_str;
        }
        else
        {
            throw std::string("No such platform pID: " + std::to_string(pID));
        }

        auto pIterator = pNames.find(desired_platform_name);
        if (pIterator != pNames.end())
        {
            std::cout << pIterator->first
                      << " " << pIterator->second << std::endl;
            pIndex = pIterator->second;
        }
        else
        {
            throw std::string(desired_platform_name + " was not found");
        }

        std::vector<cl::Device> devices;
        platforms[pIndex].getDevices(CL_DEVICE_TYPE_GPU, &devices);

        assert(dID < devices.size());

        cl::Device device = devices[dID];

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
