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
#ifndef _KERNEL_CAHCE_HPP_
#define _KERNEL_CAHCE_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include <string>
#include <map>
/**
 * @brief The KernelCache class Build and cache kernels
 * singleton
 */
class KernelCache
{

public:

    typedef std::map<unsigned int, cl::Kernel> KernelMap;

    static KernelCache& getInstance();

    static cl::Kernel get(cl::CommandQueue& queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params = "");

    const cl::Program* getProgram(cl::CommandQueue& queue,
                              const std::string& program_name,
                              const std::string& params = "");

    cl::Kernel getKernel(cl::CommandQueue &queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params = "");


private:


    unsigned int rsHash(const std::string& key);

    KernelMap kernel_map;

    KernelCache();

    static KernelCache singleton;
};

#endif //_KERNEL_CACHE_HPP_
