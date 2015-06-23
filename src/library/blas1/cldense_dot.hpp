#pragma once
#ifndef _CLSPARSE_DOT_HPP_
#define _CLSPARSE_DOT_HPP_

#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include "commons.hpp"
#include "atomic_reduce.hpp"
#include "internal/data_types/clvector.hpp"

template<typename T>
clsparseStatus
inner_product (clsparseVectorPrivate* partial,
     const clsparseVectorPrivate* pX,
     const clsparseVectorPrivate* pY,
     const cl_ulong size,
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
            + " -DN_THREADS=" + std::to_string(nthreads);

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "inner_product", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << partial->values
             << pX->values
             << pY->values;

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

template<typename T>
clsparseStatus dot(clsparseScalarPrivate* pR,
                   const clsparseVectorPrivate* pX,
                   const clsparseVectorPrivate* pY,
                   const clsparseControl control)
{

    cl_int status;

    init_scalar(pR, (T)0, control);

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */
    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    cl_ulong xSize = pX->n - pX->offset();
    cl_ulong ySize = pY->n - pY->offset();

    assert (xSize == ySize);

    cl_ulong size = xSize;


    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        clsparseVectorPrivate partial;
        clsparseInitVector(&partial);
        partial.n = REDUCE_BLOCKS_NUMBER;

        clMemRAII<T> rPartial (control->queue(), &partial.values, partial.n);

        status = inner_product<T>(&partial, pX, pY, size,  REDUCE_BLOCKS_NUMBER,
                               REDUCE_BLOCK_SIZE, control);

        if (status != clsparseSuccess)
        {
            return clsparseInvalidKernelExecution;
        }

       status = atomic_reduce<T>(pR, &partial, REDUCE_BLOCK_SIZE,
                                     control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;
}

/*
 * clsparse::array
 */
template<typename T>
clsparseStatus
inner_product (clsparse::array_base<T>& partial,
     const clsparse::array_base<T>& pX,
     const clsparse::array_base<T>& pY,
     const cl_ulong size,
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
            + " -DN_THREADS=" + std::to_string(nthreads);

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "inner_product", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << partial.data()
             << pX.data()
             << pY.data();

    cl::NDRange local(REDUCE_BLOCK_SIZE);
    cl::NDRange global(REDUCE_BLOCKS_NUMBER * REDUCE_BLOCK_SIZE);


    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

template<typename T>
clsparseStatus dot(clsparse::array_base<T>& pR,
                   const clsparse::array_base<T>& pX,
                   const clsparse::array_base<T>& pY,
                   const clsparseControl control)
{

    cl_int status;

    //not necessary to have it, but remember to init the pR with the proper value
    init_scalar(pR, (T)0, control);

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */
    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    cl_ulong xSize = pX.size();
    cl_ulong ySize = pY.size();

    assert (xSize == ySize);

    cl_ulong size = xSize;

    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        clsparse::vector<T> partial(control, REDUCE_BLOCKS_NUMBER, 0,
                                   CL_MEM_READ_WRITE, false);

        status = inner_product<T>(partial, pX, pY, size,  REDUCE_BLOCKS_NUMBER,
                               REDUCE_BLOCK_SIZE, control);

        if (status != clsparseSuccess)
        {
            return clsparseInvalidKernelExecution;
        }

       status = atomic_reduce<T>(pR, partial, REDUCE_BLOCK_SIZE,
                                     control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_DOT_HPP_
