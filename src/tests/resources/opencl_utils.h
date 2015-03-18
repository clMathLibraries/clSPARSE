#pragma once
#ifndef _OPENCL_UTILS_H_
#define _OPENCL_UTILS_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <string.h>
#include <iostream>

cl_int getPlatforms(cl_platform_id **platforms, cl_uint* num_platforms)
{
    cl_int status = CL_SUCCESS;

    status = clGetPlatformIDs (0, NULL, num_platforms);

    *platforms =
            (cl_platform_id*) malloc (*num_platforms * sizeof(cl_platform_id));

    status = clGetPlatformIDs (*num_platforms, *platforms, NULL);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with getting platfofm values" << std::endl;
        return status;
    }
    return status;
}

void printPlatforms(const cl_platform_id* platforms,
                    const cl_uint num_platforms)
{
    const char* attributeNames[5] = { "Name", "Vendor", "Version", "Profile",
                                      "Extensions" };

    const cl_platform_info attributeTypes[5] =
    { CL_PLATFORM_NAME, CL_PLATFORM_VENDOR, CL_PLATFORM_VERSION,
      CL_PLATFORM_PROFILE, CL_PLATFORM_EXTENSIONS };

    const int attributeCount = sizeof(attributeNames) / sizeof(char*);


    // for each platform print all attributes
    for (int i = 0; i < num_platforms; i++) {

        std::cout << "\n" << i+1 << ". Platform(id = " << i << ")" << std::endl;

        for (int j = 0; j < attributeCount; j++) {

            size_t infoSize;
            char* info;
            // get platform attribute value size
            clGetPlatformInfo(platforms[i], attributeTypes[j], 0, NULL, &infoSize);
            info = (char*) malloc(infoSize);

            // get platform attribute value
            clGetPlatformInfo(platforms[i], attributeTypes[j], infoSize, info, NULL);

            printf("  %d.%d %-11s: %s\n", i+1, j+1, attributeNames[j], info);

            free(info);
        }
        printf("\n");
    }

}


//get the first available device from given platform
cl_int getDevice (const cl_platform_id platform,
                cl_device_id* device,
                cl_device_type type)
{
    cl_int status;
    cl_uint num_devices = 0;

    //get count of given device types
    status = clGetDeviceIDs(platform, type, 0, NULL, &num_devices);
    if (status != CL_SUCCESS)
        return status;

    cl_device_id* devices = (cl_device_id*)
            malloc(num_devices*sizeof(cl_device_id));
    status = clGetDeviceIDs(platform, type, num_devices, devices, NULL);
    if (status != CL_SUCCESS)
        return status;

    *device = devices[0];

    return status;
}

void printDeviceInfo(const cl_device_id device)
{
    size_t size;
    char* value;

    clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &size);
    value = (char*) malloc(size);
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, value, NULL);
    std::cout << "Device Name: " << value << std::endl;
    free(value);
}

#endif //_OPEN_CL_UTILS_H_

