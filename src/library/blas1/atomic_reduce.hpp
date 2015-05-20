#pragma once
#ifndef _CLSPARSE_ATOMIC_REDUCE_HPP_
#define _CLSPARSE_ATOMIC_REDUCE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "reduce_operators.hpp"

/* Helper function used in reduce type operations
 * pR = \sum pX
 * ASSUMPTIONS:
 *      pR initial value is set
 *      pX size is equal to wg_size;
 *      wg_size is the workgroup size
*/

enum PRECISION{
    clsparseFloat = 0,
    clsparseDouble
};

template<PRECISION FPTYPE, ReduceOperator OP = DUMMY>
clsparseStatus
atomic_reduce(clsparseScalarPrivate* pR,
              const clsparseVectorPrivate* pX,
              const cl_ulong wg_size,
              const clsparseControl control)
{
    assert(wg_size == pX->n);

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -D" + ReduceOperatorTrait<OP>::operation;

    if (FPTYPE == clsparseFloat)
    {
        std::string options = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
                + " -DATOMIC_FLOAT";
        params.append(options);
    }
    else if (FPTYPE == clsparseDouble)
    {
        std::string options = std::string()
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
                + " -DATOMIC_DOUBLE";
        params.append(options);
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


#endif //_CLSPARSE_ATOMIC_REDUCE_HPP_
