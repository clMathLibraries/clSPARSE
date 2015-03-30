#pragma once
#ifndef _OPENCL_UTILS_H_
#define _OPENCL_UTILS_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <map>
#include <string>
#include <iostream>

typedef enum _platform {
    AMD = 0,
    NVIDIA
} cl_platform_type;

const static std::string amd_platform_str = "AMD Accelerated Parallel Processing";
const static std::string nvidia_platform_str = "NVIDIA CUDA";


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

cl::Device getDevice(cl_platform_type pID, cl_uint dID)
{
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

    return device;

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

    free( devices );
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

