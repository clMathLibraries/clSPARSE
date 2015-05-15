#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"

#include "commons.hpp"
#include "atomic_reduce.hpp"

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

template<typename T, PRECISION FPTYPE>
clsparseStatus dot(clsparseScalarPrivate* pR,
                   const clsparseVectorPrivate* pX,
                   const clsparseVectorPrivate* pY,
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

       status = atomic_reduce<FPTYPE>(pR, &partial, REDUCE_BLOCK_SIZE,
                                     control);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;

}

clsparseStatus
cldenseSdot (clsparseScalar* r,
             const clsparseVector* x,
             const clsparseVector* y,
             const clsparseControl control)
{
    clsparseScalarPrivate* pR = static_cast<clsparseScalarPrivate*>( r );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseVectorPrivate* pY = static_cast<const clsparseVectorPrivate*> ( y );

    return dot<cl_float, FLOAT>(pR, pX, pY, control);
}

clsparseStatus
cldenseDdot (clsparseScalar* r,
             const clsparseVector* x,
             const clsparseVector* y,
             const clsparseControl control)
{
    clsparseScalarPrivate* pDot = static_cast<clsparseScalarPrivate*>( r );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseVectorPrivate* pY = static_cast<const clsparseVectorPrivate*> ( y );

    return dot<cl_double, DOUBLE>(pDot, pX, pY, control);
}
