/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef _OPENCL_UTILS_H_
#define _OPENCL_UTILS_H_

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION

#include <CL/cl2.hpp>

#include <map>
#include <string>
#include <iostream>

typedef enum _platform {
    AMD = 0,
    NVIDIA
} cl_platform_type;

const static std::string amd_platform_str = "AMD";
const static std::string nvidia_platform_str = "NVIDIA";

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
