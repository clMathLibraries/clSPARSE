#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include "atomic_reduce.hpp"

#include <algorithm>

clsparseStatus
reduce (const clsparseVectorPrivate* pX, clsparseVectorPrivate* partialSum,
        cl_ulong REDUCE_BLOCKS_NUMBER, cl_ulong REDUCE_BLOCK_SIZE,
        const std::string& params,
        const clsparseControl control)
{

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "reduce", "reduce", params);

    KernelWrap kWrapper(kernel);

    kWrapper << (cl_ulong)pX->n
             << pX->values
             << partialSum->values;

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
cldenseSreduce(clsparseScalar *s,
               const clsparseVector *x,
               const clsparseControl control)
{
    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */

    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    {
        clMemRAII<cl_float> rSum (control->queue(), pSum->value);
        cl_float* fSum = rSum.clMapMem( CL_TRUE, CL_MAP_WRITE, pSum->offset(), 1);
        *fSum = 0.0f;
    }

    cl_int status;
    if (pX->n > 0)
    {

        cl::Context context = control->getContext();

        //vector for partial sums of X;
        clsparseVectorPrivate partialSum;
        clsparseInitVector( &partialSum );
#if (BUILD_CLVERSION < 200)
        partialSum.values = ::clCreateBuffer(context(), CL_MEM_READ_WRITE,
                                             REDUCE_BLOCKS_NUMBER * sizeof(cl_float),
                                             NULL, &status);
        if (status != CL_SUCCESS)
        {
            std::cout << "Error: Problem with allocating partialSum vector: "
                      << status << std::endl;
            return clsparseInvalidMemObj;
        }
#else
        partialSum.values = ::clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                         REDUCE_BLOCKS_NUMBER * sizeof(cl_float),
                                         0);
        if (partialSum.values == nullptr)
        {
            std::cout << "Error: Problem with allocating partialSum vector: "
                      << status << std::endl;
            return clsparseInvalidMemObj;

        }
#endif
        partialSum.n = REDUCE_BLOCKS_NUMBER;

        cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

        std::string params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = reduce(pX, &partialSum, REDUCE_BLOCKS_NUMBER, REDUCE_BLOCK_SIZE, params, control);

        if (status != CL_SUCCESS)
        {
#if (BUILD_CLVERSION < 200)
            ::clReleaseMemObject(partialSum.values);
#else
            ::clSVMFree(context(), partialSum.values)
#endif
            return clsparseInvalidKernelExecution;
        }

        status = atomic_reduce<FLOAT>(pSum, &partialSum, REDUCE_BLOCK_SIZE, control);

        // free temp data
#if (BUILD_CLVERSION < 200)
        ::clReleaseMemObject(partialSum.values);

#else
        ::clSVMFree(context(), partialSum.values)
#endif
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    } // pX->n > 0

    return clsparseSuccess;
}

clsparseStatus
cldenseDreduce(clsparseScalar *s,
               const clsparseVector *x,
               const clsparseControl control)
{
    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    // with REDUCE_BLOCKS_NUMBER = 256 final reduction can be performed
    // within one block;
    const cl_ulong REDUCE_BLOCKS_NUMBER = 256;

    /* For future optimisation
    //workgroups per compute units;
    const cl_uint  WG_PER_CU = 64;
    const cl_ulong REDUCE_BLOCKS_NUMBER = control->max_compute_units * WG_PER_CU;
    */

    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    {
        clMemRAII<cl_double> rSum (control->queue(), pSum->value);
        cl_double* fSum = rSum.clMapMem( CL_TRUE, CL_MAP_WRITE, pSum->offset(), 1);
        *fSum = 0.0;
    }


    cl_int status;
    if (pX->n > 0)
    {
        cl::Context context = control->getContext();

        //vector for partial sums of X;
        clsparseVectorPrivate partialSum;
        clsparseInitVector( &partialSum );

#if (BUILD_CLVERSION < 200)
        partialSum.values = ::clCreateBuffer(context(), CL_MEM_READ_WRITE,
                                             REDUCE_BLOCKS_NUMBER * sizeof(cl_double),
                                             NULL, &status);
#else
        partialSum.values = ::clSVMAlloc(context(), CL_MEM_READ_WRITE,
                                         REDUCE_BLOCKS_NUMBER * sizeof(cl_double),
                                         0);
#endif
        partialSum.n = REDUCE_BLOCKS_NUMBER;


        if (status != CL_SUCCESS)
        {
            return clsparseInvalidMemObj;
        }

        cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

        std::string params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        status = reduce(pX, &partialSum, REDUCE_BLOCKS_NUMBER, REDUCE_BLOCK_SIZE, params, control);

        if (status != CL_SUCCESS)
        {
#if (BUILD_CLVERSION < 200)
            ::clReleaseMemObject(partialSum.values);

#else
            ::clSVMFree(context(), partialSum.values)
        #endif
            return clsparseInvalidKernelExecution;
        }

        status = atomic_reduce<DOUBLE>(pSum, &partialSum, REDUCE_BLOCK_SIZE, control);

        // free temp data
#if (BUILD_CLVERSION < 200)
        ::clReleaseMemObject(partialSum.values);

#else
        ::clSVMFree(context(), partialSum.values)
        #endif
        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }
    } // pX->n > 0

    return clsparseSuccess;
}
