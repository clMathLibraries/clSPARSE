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
#ifndef _KERNEL_WRAP_HPP_
#define _KERNEL_WRAP_HPP_

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION

#include <CL/cl2.hpp>

#include <iostream>
#include <cassert>
#include <vector>
#include <type_traits>

#include "clSPARSE.h"
#include "clsparse-control.hpp"
#include "ocl-type-traits.hpp"

//! \brief Class interface for specifying NDRange values.
//! not to include cl.hpp this is moved here and implementation
//! is changed to std::vector
class KernelWrap
{
public:
    KernelWrap(cl::Kernel &kernel);

    cl_int run (clsparseControl control, const cl::NDRange global,
                const cl::NDRange local);

    void reset()
    {
        argCounter = 0;
    }

    template<typename T,
             typename std::enable_if<!is_pointer_fundamental<T>::value>::type* = nullptr>
    KernelWrap& operator<< (const T& arg)
    {
        //std::cout << "NOT IMPLEMENTED" << std::endl;
        static_assert(!is_pointer_fundamental<T>::value,
                      "!is_pointer_fundamental<T>::value>");

        return *this;
    }

    template<typename T,
             typename std::enable_if<is_pointer_fundamental<T>::value>::type* = nullptr>
    KernelWrap& operator<< (const T arg)
    {
        std::cout << "Void* types should be managed here for CL20" << std::endl;
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());

        int status = clSetKernelArgSVMPointer(kernel(),
                                              argCounter++, arg );

        return *this;
    }

    inline KernelWrap& operator<<(const cl_mem& val)
    {
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());

        cl_int status = 0;
        status = kernel.setArg(argCounter++, sizeof(cl_mem), (const void*)val);


        return *this;
    }

    // support for cl::LocalSpaceArg
    inline KernelWrap& operator<<(const cl::LocalSpaceArg& local)
    {
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());
        //there is no getArgInfo in OpenCL 1.1
#if defined(CL_VERSION_1_2)
        assert(kernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(argCounter)
               == CL_KERNEL_ARG_ADDRESS_LOCAL);
#endif

        kernel.setArg(argCounter++, local);
        return *this;
    }

    // support for raw cl::Buffer
    inline KernelWrap& operator<<(const cl::Buffer& buffer)
    {
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());
        kernel.setArg(argCounter++, buffer);
        return *this;
    }

private:

    cl::Kernel kernel;
    cl_uint argCounter;
    cl_uint addrBits;

};



// support for basic types

//cl 1.1 //there is no getArgInfo in OpenCL 1.1
//kind of leap of faith if on OpenCL everyting is ok with setting the args.
#if (BUILD_CLVERSION == 110)
#define KERNEL_ARG_BASE_TYPE(TYPE, TYPE_STRING) \
        template<> inline KernelWrap& \
        KernelWrap::operator<< <TYPE>(const TYPE& arg) \
        { \
            assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>()); \
            kernel.setArg(argCounter++, arg); \
            return *this; \
        }

#elif (BUILD_CLVERSION == 120)

// This string.compare gets around malformed null-terminating strings returned on Nvidia platforms
#define KERNEL_ARG_BASE_TYPE(TYPE, TYPE_STRING) \
        template<> inline KernelWrap& \
        KernelWrap::operator<< <TYPE>(const TYPE& arg) \
        { \
            assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>()); \
            assert(kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(argCounter).compare( 0, sizeof(TYPE_STRING)-1, TYPE_STRING, 0, sizeof(TYPE_STRING)-1 ) == 0 ); \
            kernel.setArg(argCounter++, arg); \
            return *this; \
        }
#else // (BUILD_CLVERSION == 200)
// This string.compare gets around malformed null-terminating strings returned on Nvidia platforms
#define KERNEL_ARG_BASE_TYPE(TYPE, TYPE_STRING) \
        template<> inline KernelWrap& \
        KernelWrap::operator<< <TYPE>(const TYPE& arg) \
        { \
            assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>()); \
            assert(kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(argCounter).compare( 0, sizeof(TYPE_STRING)-1, TYPE_STRING, 0, sizeof(TYPE_STRING)-1 ) == 0 ); \
            int status =  clSetKernelArgSVMPointer(kernel(), \
                          argCounter++, \
                          (void *)(&arg)); \
            return *this; \
        }
#endif

KERNEL_ARG_BASE_TYPE( cl_int, "int" )
KERNEL_ARG_BASE_TYPE( cl_uint, "uint" )
KERNEL_ARG_BASE_TYPE( cl_short, "short" )
KERNEL_ARG_BASE_TYPE( cl_ushort, "ushort" )
KERNEL_ARG_BASE_TYPE( cl_float, "float" )
KERNEL_ARG_BASE_TYPE( cl_double, "double" )
KERNEL_ARG_BASE_TYPE( cl_ulong, "ulong" )


#endif //_KERNEL_WRAP_HPP_
