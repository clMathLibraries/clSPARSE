#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

clsparseStatus
dot (clsparseVectorPrivate* partial,
     const clsparseVectorPrivate* pX,
     const clsparseVectorPrivate* pY,
     const cl_ulong size,
     const cl_ulong REDUCE_BLOCKS_NUMBER,
     const cl_ulong REDUCE_BLOCK_SIZE,
     const std::string& params,
     const clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "dot_block", params);

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

clsparseStatus
dot_final (clsparseScalarPrivate* pR,
           const clsparseVectorPrivate* pX,
           const cl_ulong group_size,
           const std::string& params,
           const clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "dot", "dot_final", params);

    KernelWrap kWrapper(kernel);
    kWrapper << (cl_ulong)pX->n
             << pX->values
             << pR->value;

    int blocksNum = (pX->n + group_size - 1) / group_size;
    int globalSize = blocksNum * group_size;

    cl::NDRange local(group_size);
    cl::NDRange global(globalSize);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}


clsparseStatus
cldenseSdot (clsparseScalar* r,
             const clsparseVector* x,
             const clsparseVector* y,
             const clsparseControl control)
{
    clsparseScalarPrivate* pDot = static_cast<clsparseScalarPrivate*>( r );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );
    const clsparseVectorPrivate* pY = static_cast<const clsparseVectorPrivate*> ( y );

    {
        clMemRAII<cl_float> rDot (control->queue(), pDot->value);
        cl_float* fDot = rDot.clMapMem( CL_TRUE, CL_MAP_WRITE, pDot->offset(), 1);
        *fDot = 0.0f;
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

    cl_ulong xSize = pX->n - pX->offset();
    cl_ulong ySize = pY->n - pY->offset();

    assert (xSize == ySize);

    cl_ulong size = xSize;

    cl_int status;

    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        clsparseVectorPrivate partialDot;
        clsparseInitVector (&partialDot);

#if (BUILD_CLVERSION < 200)
        partialDot.values = ::clCreateBuffer(context(), CL_MEM_READ_WRITE,
                                             REDUCE_BLOCKS_NUMBER * sizeof(cl_float),
                                             NULL, &status);
        if (status != CL_SUCCESS)
        {
            std::cout << "Error: Problem with allocating paritalDot vector: "
                      << status << std::endl;
            return clsparseInvalidMemObject;
        }
#else
        partialDot.values = ::clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                         REDUCE_BLOCKS_NUMBER * sizeof(cl_float),
                                         0);
        if (partialDot.values == nullptr)
        {
            std::cout << "Error: Problem with allocating partialDot vector: "
                      << status << std::endl;
            return clsparseInvalidMemObj;
        }
#endif

        partialDot.n  = REDUCE_BLOCKS_NUMBER;

        cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

        std::string params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = dot (&partialDot, pX, pY, size,
                      REDUCE_BLOCKS_NUMBER, REDUCE_BLOCK_SIZE,
                      params, control);

        if (status != clsparseSuccess)
        {
#if (BUILD_CLVERSION < 200)
            ::clReleaseMemObject(partialDot.values);
#else
            ::clSVMFree(context(), partialDot.values)
#endif
            return clsparseInvalidKernelExecution;
        }

        params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
                + " -DATOMIC_FLOAT"
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                // not used but necessary to have to compile the program.
                // I dont want to create new file for this simple kernel;
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = dot_final(pDot, &partialDot, REDUCE_BLOCK_SIZE, params, control);

        // free temp data
#if (BUILD_CLVERSION < 200)
        ::clReleaseMemObject(partialDot.values);
#else
        ::clSVMFree(context(), partialDot.values)
#endif
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;

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

    {
        clMemRAII<cl_double> rDot (control->queue(), pDot->value);
        cl_double* fDot = rDot.clMapMem( CL_TRUE, CL_MAP_WRITE, pDot->offset(), 1);
        *fDot = 0.0;
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

    cl_ulong xSize = pX->n - pX->offset();
    cl_ulong ySize = pY->n - pY->offset();

    assert (xSize == ySize);

    cl_ulong size = xSize;

    cl_int status;

    if (size > 0)
    {
        cl::Context context = control->getContext();

        //partial result
        clsparseVectorPrivate partialDot;
        clsparseInitVector (&partialDot);

#if (BUILD_CLVERSION < 200)
        partialDot.values = ::clCreateBuffer(context(), CL_MEM_READ_WRITE,
                                             REDUCE_BLOCKS_NUMBER * sizeof(cl_double),
                                             NULL, &status);
        if (status != CL_SUCCESS)
        {
            std::cout << "Error: Problem with allocating paritalDot vector: "
                      << status << std::endl;
            return clsparseInvalidMemObject;
        }
#else
        partialDot.values = ::clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                         REDUCE_BLOCKS_NUMBER * sizeof(cl_double),
                                         0);
        if (partialDot.values == nullptr)
        {
            std::cout << "Error: Problem with allocating partialDot vector: "
                      << status << std::endl;
            return clsparseInvalidMemObj;
        }
#endif

        partialDot.n  = REDUCE_BLOCKS_NUMBER;

        cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

        std::string params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = dot (&partialDot, pX, pY, size,
                      REDUCE_BLOCKS_NUMBER, REDUCE_BLOCK_SIZE,
                      params, control);

        if (status != clsparseSuccess)
        {
#if (BUILD_CLVERSION < 200)
            ::clReleaseMemObject(partialDot.values);
#else
            ::clSVMFree(context(), partialDot.values)
#endif
            return clsparseInvalidKernelExecution;
        }

        params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
                + " -DATOMIC_DOUBLE"
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                // not used but necessary to have to compile the program.
                // I dont want to create new file for this simple kernel;
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = dot_final(pDot, &partialDot, REDUCE_BLOCK_SIZE, params, control);

        // free temp data
#if (BUILD_CLVERSION < 200)
        ::clReleaseMemObject(partialDot.values);
#else
        ::clSVMFree(context(), partialDot.values)
#endif
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    }

    return clsparseSuccess;
}
