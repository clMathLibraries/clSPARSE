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
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void axpy(const SIZE_TYPE pXsize,
          __global VALUE_TYPE* pY,
          const SIZE_TYPE pYOffset,
          __global const VALUE_TYPE* pAlpha,
         const SIZE_TYPE pAlphaOffset,
          __global const VALUE_TYPE* pX,
          const SIZE_TYPE pXOffset)
{

    const int index = get_global_id(0);

    if (index >= pXsize) return;

    const VALUE_TYPE alpha = *(pAlpha + pAlphaOffset);

    pY[index + pYOffset] = alpha * pX[index + pXOffset] + pY[index + pYOffset];
}

)"
