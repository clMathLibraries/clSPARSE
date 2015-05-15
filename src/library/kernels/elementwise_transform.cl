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
)"

R"(
VALUE_TYPE operation(VALUE_TYPE A, VALUE_TYPE B)
{
#ifdef OP_PLUS
    return A + B;
#elif OP_MINUS
    return A - B;
#elif OP_MULTIPLY
    return A * B;
#else
    return 0;
#endif
}
)"



R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void transform (const SIZE_TYPE size,
                __global VALUE_TYPE* pR,
                __global const VALUE_TYPE* pX,
                __global const VALUE_TYPE* pY)
{
    const int index = get_global_id(0);

    if (index >= size) return;

    pR[index] = operaton(x[index], y[index]);

}

)"
