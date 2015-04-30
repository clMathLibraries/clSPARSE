#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include <algorithm>

clsparseStatus
reduce (const clsparseVectorPrivate* pX, clsparseVectorPrivate* partialSum,
        cl_ulong REDUCE_BLOCKS_NUMBER, cl_ulong REDUCE_BLOCK_SIZE,
        const std::string& params,
        const clsparseControl control)
{

    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "reduce", "reduce_block", params);

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
clsparseSreduce(clsparseScalar *sum,
                const clsparseVector *x,
                const clsparseControl control)
{
    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( sum );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    const cl_ulong REDUCE_BLOCKS_NUMBER = 32;
    const cl_ulong REDUCE_BLOCK_SIZE = 256;

    clMemRAII<cl_float> rSum (control->queue(), pSum->value);
    cl_float* fSum = rSum.clMapMem( CL_TRUE, CL_MAP_WRITE, pSum->offset(), 1);
    *fSum = 0.0f;

    cl_int status;
    if (pX->n > 0)
    {

        cl::Context context = control->getContext();

        //vector for partial sums of X;
        clsparseVectorPrivate partialSum;
        clsparseInitVector( &partialSum );
        partialSum.values = ::clCreateBuffer(context(), CL_MEM_READ_WRITE,
                                              REDUCE_BLOCKS_NUMBER * sizeof(cl_float),
                                              NULL, &status);
        partialSum.n = REDUCE_BLOCKS_NUMBER;
        clMemRAII<cl_float> rPartialSum (control->queue(), partialSum.values);


        if (status != CL_SUCCESS)
        {
            return clsparseInvalidMemObj;
        }

        cl_ulong nthreads = REDUCE_BLOCK_SIZE * REDUCE_BLOCKS_NUMBER;

        std::string params = std::string()
                + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
                + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
                + " -DWG_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DREDUCE_BLOCK_SIZE=" + std::to_string(REDUCE_BLOCK_SIZE)
                + " -DN_THREADS=" + std::to_string(nthreads);

        reduce(pX, &partialSum, REDUCE_BLOCKS_NUMBER, REDUCE_BLOCK_SIZE, params, control);

        cl_float* hPartialSum = rPartialSum.clMapMem(CL_TRUE, CL_MAP_READ, partialSum.offset(), partialSum.n);

        *fSum = std::accumulate(hPartialSum, hPartialSum + partialSum.n, 0.0f);

    }

}

clsparseStatus
clsparseDreduce(clsparseScalar *sum,
                const clsparseVector *x,
                const clsparseControl control)
{

}
