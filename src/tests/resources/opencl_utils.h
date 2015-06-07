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

cl_int getPlatforms( cl_platform_id **platforms, cl_uint* num_platforms );

void printPlatforms( const cl_platform_id* platforms,
                     const cl_uint num_platforms );

cl::Device getDevice( cl_platform_type pID, cl_uint dID );

//get the first available device from given platform
cl_int getDevice( const cl_platform_id platform,
                  cl_device_id* device,
                  cl_device_type type );

void printDeviceInfo( const cl_device_id device );
#endif //_OPEN_CL_UTILS_H_

