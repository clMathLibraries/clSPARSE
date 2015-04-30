R"(
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

#ifndef SIZE_TYPE
#error SIZE_TYPE undefined!
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif

#ifndef REDUCE_BLOCK_SIZE
#error REDUCE_BLOCK_SIZE undefined!
#endif

#ifndef N_THREADS
#error N_THREADS undefined!
#endif

__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void reduce_block (const SIZE_TYPE size,
          __global const VALUE_TYPE* pX,
          __global VALUE_TYPE* pSum)
{
    __local VALUE_TYPE buf_tmp[REDUCE_BLOCK_SIZE];

    SIZE_TYPE idx = get_global_id(0);

    SIZE_TYPE block_idx = idx / REDUCE_BLOCK_SIZE;
    SIZE_TYPE thread_in_block_idx = idx % REDUCE_BLOCK_SIZE;

    SIZE_TYPE eidx = idx;

    VALUE_TYPE sum = 0;
    while(eidx < size)
    {
        sum += pX[eidx];
        eidx += N_THREADS;
    }

    buf_tmp[thread_in_block_idx] = sum;

    barrier(CLK_LOCAL_MEM_FENCE);

    // TODO: Seqential part
    if (get_local_id(0) == 0)
    {
        sum = 0.0;
        for (uint i = 0; i < REDUCE_BLOCK_SIZE; i++)
        {
            sum += buf_tmp[i];
        }

        pSum[ block_idx ] = sum;
    }
}
)"
