#pragma once
#ifndef _CLSPARSE_COMMONS_HPP_
#define _CLSPARSE_COMMONS_HPP_

#include "include/clSPARSE-private.hpp"

template <typename T>
inline void init_scalar(clsparseScalarPrivate* scalar, T value,
                        const clsparseControl control)
{
    clMemRAII<T> rScalar (control->queue(), scalar->value);
    T* fR = rScalar.clMapMem( CL_TRUE, CL_MAP_WRITE, scalar->offset(), 1);
    *fR  = value;
}

template <typename T>
inline clsparseStatus
allocateVector(clsparseVectorPrivate* vector, cl_ulong size,
               const clsparseControl control)
{
    cl_int status;

    clsparseInitVector( vector );

#if (BUILD_CLVERSION < 200)
    vector->values = ::clCreateBuffer(control->getContext()(),
                                      CL_MEM_READ_WRITE, size * sizeof(T),
                                      NULL, &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Error: Problem with allocating partialSum vector: "
                  << status << std::endl;
        return clsparseInvalidMemObj;
    }
#else
    vector.values = ::clSVMAlloc(control->getContext()(), CL_MEM_READ_WRITE,
                                 size * sizeof(T), 0);
    if (vector.values == nullptr)
    {
        std::cout << "Error: Problem with allocating partialSum vector: "
                  << status << std::endl;
        return clsparseInvalidMemObj;

    }
#endif
    vector->n = size;
    return clsparseSuccess;
}

inline void releaseVector(clsparseVectorPrivate* vector,
                          const clsparseControl control)
{
#if (BUILD_CLVERSION < 200)
    ::clReleaseMemObject(vector->values);
#else
    ::clSVMFree(control->getContext()(), vector.values)
#endif
}

#endif //_CLSPARSE_COMMONS_HPP_
