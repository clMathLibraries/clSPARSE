#pragma once
#ifndef _CLSPARSE_ELEMENTWISE_HPP_
#define _CLSPARSE_ELEMENTWISE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"

#include "elementwise_operators.hpp"

#include "internal/data_types/clarray.hpp"



/* Elementwise operation on two vectors
*/

template<typename T, ElementWiseOperator OP>
clsparseStatus
elementwise_transform(clsparseVectorPrivate* r,
                      const clsparseVectorPrivate* x,
                      const clsparseVectorPrivate* y,
                      clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    assert(x->n == y->n);
    assert(x->n == r->n);

    cl_ulong size = x->n;
    cl_uint wg_size = 256;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "elementwise_transform",
                                         "transform", params);

    KernelWrap kWrapper (kernel);

    kWrapper << size << r->values << x->values << y->values;

    int blocks = (size + wg_size - 1) / wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(blocks * wg_size);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

/*
 * clsparse::array
 */
template<typename T, ElementWiseOperator OP>
clsparseStatus
elementwise_transform(clsparse::array<T>& r,
                      const clsparse::array<T>& x,
                      const clsparse::array<T>& y,
                      clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    assert(x.size() == y.size());
    assert(x.size() == r.size());

    cl_ulong size = x.size();
    cl_uint wg_size = 256;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "elementwise_transform",
                                         "transform", params);

    KernelWrap kWrapper (kernel);

    kWrapper << size << r.buffer() << x.buffer() << y.buffer();

    int blocks = (size + wg_size - 1) / wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(blocks * wg_size);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_ELEMENTWISE_HPP_
