R"(
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

#ifndef SIZE_TYPE
#error SIZE_TYPE undefined!
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif


void atomic_add_float (global VALUE_TYPE *ptr, VALUE_TYPE temp)
{
#ifdef ATOMIC_DOUBLE
        unsigned long newVal;
        unsigned long prevVal;
        do
        {
                prevVal = as_ulong(*ptr);
                newVal = as_ulong(temp + *ptr);
        } while (atom_cmpxchg((global unsigned long *)ptr, prevVal, newVal) != prevVal);

#elif ATOMIC_FLOAT
        unsigned int newVal;
        unsigned int prevVal;
        do
        {
                prevVal = as_uint(*ptr);
                newVal = as_uint(temp + *ptr);
        } while (atomic_cmpxchg((global unsigned int *)ptr, prevVal, newVal) != prevVal);
#endif
}
)"

R"(
VALUE_TYPE operation(VALUE_TYPE A)
{
#ifdef OP_RO_SQRT
    return sqrt(A);
#else
    return A;
#endif
}
)"

R"(
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
__kernel
void reduce_block (__global VALUE_TYPE* pSum,
                   __global const VALUE_TYPE* pX)
{
    SIZE_TYPE idx = get_global_id(0);

    atomic_add_float(&pSum[0], pX[idx]);

    barrier(CLK_LOCAL_MEM_FENCE);

    //at the end we want to modify the final value like in L2 norm.
    pSum[0] = operation(pSum[0]);
}
)"
