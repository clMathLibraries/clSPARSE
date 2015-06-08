#pragma once
#ifndef _CLSPARSE_AXPBY_HPP_
#define _CLSPARSE_AXPBY_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"

#include "blas1/elementwise_operators.hpp"

#include "internal/data_types/clvector.hpp"

template<typename T, ElementWiseOperator OP = EW_PLUS>
clsparseStatus
axpby(cl_ulong size,
      clsparseVectorPrivate* pY,
      const clsparseScalarPrivate* pAlpha,
      const clsparseVectorPrivate* pX,
      const clsparseScalarPrivate* pBeta,
      const clsparseControl control)
{

    const int group_size = 256; // this or higher? control->max_wg_size?

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string( group_size )
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "blas1", "axpby",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pY->values
             << pY->offset()
             << pAlpha->value
             << pAlpha->offset()
             << pX->values
             << pX->offset()
             << pBeta->value
             << pBeta->offset()
             << pY->values
             << pY->offset();

    int blocksNum = (size + group_size - 1) / group_size;
    int globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global (globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


//version for clsparse::array

// pY is a result container;
// y = alpha * x + beta * z;
// if z == y we have standard axpby; should we adopt the clSPARSE.h to this interface?

template<typename T, ElementWiseOperator OP = EW_PLUS>
clsparseStatus
axpby(clsparse::array_base<T>& pY,
      const clsparse::array_base<T>& pAlpha,
      const clsparse::array_base<T>& pX,
      const clsparse::array_base<T>& pBeta,
      const clsparse::array_base<T>& pZ,
      const clsparseControl control)
{

    const int group_size = 256; // this or higher? control->max_wg_size?

    const std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string( group_size )
            + " -D" + ElementWiseOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue, "blas1", "axpby",
                                         params);

    KernelWrap kWrapper(kernel);

    cl_ulong size = pY.size();

    //clsparse do not support offset;
    cl_ulong offset = 0;

    kWrapper << size
             << pY.data()
             << offset
             << pAlpha.data()
             << offset
             << pX.data()
             << offset
             << pBeta.data()
             << offset
             << pZ.data()
             << offset;

    int blocksNum = (size + group_size - 1) / group_size;
    int globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global (globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


#endif //_CLSPARSE_AXPBY_HPP_
