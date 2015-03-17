#ifndef _KERNEL_WRAP_HPP_
#define _KERNEL_WRAP_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <iostream>
#include <cassert>
#include <vector>

#include <type_traits>

#include "ocl_type_traits.hpp"

//! \brief Class interface for specifying NDRange values.
//! not to include cl.hpp this is moved here and implementation
//! is changed to std::vector
class NDRange
{
private:
    cl_uint dimensions_;

    std::vector<size_t> sizes_;
    //size_t<3> sizes_;

public:
    //! \brief Default constructor - resulting range has zero dimensions.
    NDRange()
        : dimensions_(0), sizes_(dimensions_)
    { }

    //! \brief Constructs one-dimensional range.
    NDRange(size_t size0)
        : dimensions_(1), sizes_(dimensions_)
    {
        sizes_[0] = size0;
    }

    //! \brief Constructs two-dimensional range.
    NDRange(::size_t size0, ::size_t size1)
        : dimensions_(2), sizes_(dimensions_)
    {
        sizes_[0] = size0;
        sizes_[1] = size1;
    }

    //! \brief Constructs three-dimensional range.
    NDRange(::size_t size0, ::size_t size1, ::size_t size2)
        : dimensions_(3), sizes_(dimensions_)
    {
        sizes_[0] = size0;
        sizes_[1] = size1;
        sizes_[2] = size2;
    }

    /*! \brief Conversion operator to const ::size_t *.
     *
     *  \returns a pointer to the size of the first dimension.
     */
    operator const size_t*() const {
        return (const size_t*) sizes_.data();
    }

    //! \brief Queries the number of dimensions in the range.
    cl_uint dimensions() const { return dimensions_; }
};

class KernelWrap
{
public:
    KernelWrap(cl_kernel &kernel);


    cl_int run (cl_command_queue& queue, const NDRange global,
                const NDRange local,
                const std::vector<cl_event>* events = nullptr,
                cl_event* event = nullptr);

    void reset()
    {
        argCounter = 0;
    }


    //raw buffer

    template<typename T>
    KernelWrap& operator<< (const T& arg)
    {
        std::cout << "NOT IMPLEMENTED" << std::endl;
        return *this;
    }

    inline KernelWrap& operator<<(const cl_mem& val)
    {

        assert(argCounter < numArgs);
        cl_int status =
                clSetKernelArg(*kernel, argCounter++, sizeof(cl_mem), &val);

        if (status != CL_SUCCESS)
        {
            std::cout << "Problem with setting arg num ["
                      << argCounter <<  "] into the kernel" << std::endl;
        }

        return *this;
    }




private:

    cl_kernel* kernel;
    int argCounter;
    int numArgs;

};



/** support for basic types */
#define KERNEL_ARG_BASE_TYPE(TYPE, STRING_TYPE) \
        template<> inline KernelWrap& KernelWrap::operator << (const TYPE& arg) \
        { \
            assert(argCounter < numArgs); \
          \
            void* argName = nullptr; \
            size_t size;             \
            cl_int status = clGetKernelArgInfo(*(this->kernel), argCounter, \
                                    CL_KERNEL_ARG_TYPE_NAME, 0, NULL, &size);  \
            if (status != CL_SUCCESS)                                          \
                std::cout << "obtaining size failed" << std::endl;             \
                                                                               \
            argName = malloc(size);                                            \
                                                                               \
            status = clGetKernelArgInfo(*(this->kernel), argCounter, \
                                  CL_KERNEL_ARG_TYPE_NAME,  size, argName, NULL);  \
            if (status != CL_SUCCESS)   \
                    std::cout << "obtaining info failed" << std::endl;      \
\
            std::string strArg((char*)argName);\
            int cmp = strArg.compare(STRING_TYPE); \
            assert(cmp == 0); \
            \
            status =\
                clSetKernelArg(*kernel, argCounter++, sizeof(TYPE), &arg);\
\
        if (status != CL_SUCCESS) \
        {\
            std::cout << "Problem with setting arg num ["   \
                      << argCounter <<  "] into the kernel" << std::endl; \
        }\
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
