#ifndef _KERNEL_WRAP_HPP_
#define _KERNEL_WRAP_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include <iostream>
#include <cassert>
#include <vector>
#include <type_traits>

#include "clSPARSE.h"
#include "clsparse_control.hpp"
#include "ocl_type_traits.hpp"

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

    template<typename T>
    KernelWrap& operator<< (const T& arg)
    {
        std::cout << "NOT IMPLEMENTED" << std::endl;
        return *this;
    }

    inline KernelWrap& operator<<(const cl_mem& val)
    {
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());
        kernel.setArg(argCounter++, val);

        return *this;
    }

    // support for cl::LocalSpaceArg
    inline KernelWrap& operator<<(const cl::LocalSpaceArg& local)
    {
        assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>());
        assert(kernel.getArgInfo<CL_KERNEL_ARG_ADDRESS_QUALIFIER>(argCounter)
               == CL_KERNEL_ARG_ADDRESS_LOCAL);

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
    int argCounter;

};



// support for basic types
#define KERNEL_ARG_BASE_TYPE(TYPE, TYPE_STRING) \
        template<> inline KernelWrap& \
        KernelWrap::operator<< <TYPE>(const TYPE& arg) \
        { \
            assert(argCounter < kernel.getInfo<CL_KERNEL_NUM_ARGS>()); \
            assert(kernel.getArgInfo<CL_KERNEL_ARG_TYPE_NAME>(argCounter) \
                                                         == TYPE_STRING); \
            kernel.setArg(argCounter++, arg); \
            return *this; \
        }


KERNEL_ARG_BASE_TYPE(int, "int")
KERNEL_ARG_BASE_TYPE(unsigned int, "uint")
KERNEL_ARG_BASE_TYPE(short, "short")
KERNEL_ARG_BASE_TYPE(unsigned short, "ushort")
KERNEL_ARG_BASE_TYPE(float, "float")
KERNEL_ARG_BASE_TYPE(double, "double")
KERNEL_ARG_BASE_TYPE(long unsigned int, "ulong")


#endif //_KERNEL_WRAP_HPP_
