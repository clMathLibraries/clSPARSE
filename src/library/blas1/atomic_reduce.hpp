#pragma once
#ifndef _CLSPARSE_ATOMIC_REDUCE_HPP_
#define _CLSPARSE_ATOMIC_REDUCE_HPP_

#include <typeinfo>

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "reduce_operators.hpp"
#include "internal/data_types/clarray_base.hpp"

/* Helper function used in reduce type operations
 * pR = \sum pX
 * ASSUMPTIONS:
 *      pR initial value is set
 *      pX size is equal to wg_size;
 *      wg_size is the workgroup size
*/
template<typename T, ReduceOperator OP = RO_DUMMY>
clsparseStatus
atomic_reduce(clsparseScalarPrivate* pR,
              const clsparseVectorPrivate* pX,
              const cl_ulong wg_size,
              const clsparseControl control)
{
    assert(wg_size == pX->n);

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ReduceOperatorTrait<OP>::operation;

    if (typeid(cl_float) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_FLOAT";
        params.append(options);
    }
    else if (typeid(cl_double) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_DOUBLE";
        params.append(options);
    }
    else if (typeid(cl_int) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_INT";
        params.append(options);
    }
    else
    {
        return clsparseInvalidType;
    }

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "atomic_reduce", "reduce_block",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pR->value;
    kWrapper << pX->values;

    int blocksNum = (pX->n + wg_size - 1) / wg_size;
    int globalSize = blocksNum * wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(globalSize);

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
template<typename T, ReduceOperator OP = RO_DUMMY>
clsparseStatus
atomic_reduce(clsparse::array_base<T>& pR,
              const clsparse::array_base<T>& pX,
              const cl_ulong wg_size,
              const clsparseControl control)
{
    assert(wg_size == pX.size());

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ReduceOperatorTrait<OP>::operation;

    if (typeid(cl_float) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_FLOAT";
        params.append(options);
    }
    else if (typeid(cl_double) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_DOUBLE";
        params.append(options);
    }
    else if (typeid(cl_int) == typeid(T))
    {
        std::string options = std::string() + " -DATOMIC_INT";
        params.append(options);
    }
    else
    {
        return clsparseInvalidType;
    }

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "atomic_reduce", "reduce_block",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pR.data();
    kWrapper << pX.data();

    int blocksNum = (pX.size() + wg_size - 1) / wg_size;
    int globalSize = blocksNum * wg_size;

    cl::NDRange local(wg_size);
    cl::NDRange global(globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


#endif //_CLSPARSE_ATOMIC_REDUCE_HPP_
