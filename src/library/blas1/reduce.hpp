#pragma once
#ifndef _CLSPARSE_REDUCE_HPP_
#define _CLSPARSE_REDUCE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"

#include "commons.hpp"
#include "reduce_operators.hpp"
#include "atomic_reduce.hpp"

#include <algorithm>

#include "internal/data_types/clarray.hpp"

template <typename T, ReduceOperator OP>
clsparseStatus
global_reduce (clsparseVectorPrivate* partial,
               const clsparseVectorPrivate* pX,
               const cl_ulong REDUCE_BLOCKS_NUMBER,
               const cl_ulong REDUCE_BLOCK_SIZE,
               const clsparseControl control)
{
    cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DN_THREADS=" + std::to_string(nthreads)
            + " -D" + ReduceOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "reduce", "reduce", params);

    KernelWrap kWrapper(kernel);

    kWrapper << (cl_ulong)pX->n
             << pX->values
             << partial->values;

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

// G_OP: Global reduce operation
// F_OP: Final reduce operation, modifies final result of the reduce operation
template<typename T, ReduceOperator G_OP, ReduceOperator F_OP = RO_DUMMY>
clsparseStatus
reduce(clsparseScalarPrivate* pR,
       const clsparseVectorPrivate* pX,
       const clsparseControl control)
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

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */

    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    init_scalar(pR, (T)0.0, control);


    cl_int status;
    if (pX->n > 0)
    {
        cl::Context context = control->getContext();

        //vector for partial sums of X;
        //partial result
        clsparseVectorPrivate partial;
        clsparseInitVector(&partial);
        partial.n = REDUCE_BLOCKS_NUMBER;

        //partial will be deleted according to this object lifetime
        clMemRAII<T> rPartial (control->queue(), &partial.values, partial.n);


        status = global_reduce<T, G_OP>(&partial, pX, REDUCE_BLOCKS_NUMBER,
                                      REDUCE_BLOCK_SIZE, control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        clsparseStatus clsp_status =
                atomic_reduce<T, F_OP>(pR, &partial, REDUCE_BLOCK_SIZE, control);


        if (clsp_status!= CL_SUCCESS)
        {
            return clsp_status;
        }
    }

    return clsparseSuccess;
}

/*
 * clsparse::array version
 */

template <typename T, ReduceOperator OP>
clsparseStatus
global_reduce (clsparse::array<T>& partial,
               const clsparse::array<T>& pX,
               const cl_ulong REDUCE_BLOCKS_NUMBER,
               const cl_ulong REDUCE_BLOCK_SIZE,
               const clsparseControl control)
{
    cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
            + " -DN_THREADS=" + std::to_string(nthreads)
            + " -D" + ReduceOperatorTrait<OP>::operation;

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "reduce", "reduce", params);

    KernelWrap kWrapper(kernel);

    cl_ulong size = pX.size();

    kWrapper << size
             << pX.buffer()
             << partial.buffer();

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

// G_OP: Global reduce operation
// F_OP: Final reduce operation, modifies final result of the reduce operation
template<typename T, ReduceOperator G_OP, ReduceOperator F_OP = RO_DUMMY>
clsparseStatus
reduce(clsparse::array<T>& pR,
       const clsparse::array<T>& pX,
       const clsparseControl control)
{


    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */
    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    init_scalar(pR, (T)0, control);

    cl_int status;
    if (pX.size() > 0)
    {
        cl::Context context = control->getContext();

        //vector for partial sums of X;
        clsparse::array<T> partial(control, REDUCE_BLOCKS_NUMBER, 0,
                                   CL_MEM_READ_WRITE, false);

        status = global_reduce<T, G_OP>(partial, pX, REDUCE_BLOCKS_NUMBER,
                                      REDUCE_BLOCK_SIZE, control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        clsparseStatus clsp_status =
                atomic_reduce<T, F_OP>(pR, partial, REDUCE_BLOCK_SIZE, control);


        if (clsp_status!= CL_SUCCESS)
        {
            return clsp_status;
        }
    }

    return clsparseSuccess;
}


#endif //_CLSPARSE_REDUCE_HPP_
