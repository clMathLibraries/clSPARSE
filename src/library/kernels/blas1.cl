R"(
/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

// No reason to include these beyond version 1.2, where double is not an extension.
#if defined(DOUBLE) && __OPENCL_VERSION__ < CL_VERSION_1_2
  #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #else
    #error "Double precision floating point not supported by OpenCL implementation."
  #endif
#endif

#ifndef VALUE_TYPE
#error "VALUE_TYPE undefined!"
#endif

#ifndef SIZE_TYPE
#error "SIZE_TYPE undefined!"
#endif

#ifndef WG_SIZE
#error "WG_SIZE undefined!"
#endif
)"

R"(
VALUE_TYPE operation(VALUE_TYPE A, VALUE_TYPE B)
{
#ifdef OP_EW_PLUS
    return A + B;
#elif OP_EW_MINUS
    return A - B;
#elif OP_EW_MULTIPLY
    return A * B;
#else
    return 0;
#endif
}
)"


R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void axpy(const SIZE_TYPE size,
          __global VALUE_TYPE* pY,
          const SIZE_TYPE pYOffset,
          __global const VALUE_TYPE* pAlpha,
         const SIZE_TYPE pAlphaOffset,
          __global const VALUE_TYPE* pX,
          const SIZE_TYPE pXOffset,
          __global const VALUE_TYPE* pZ,
          const SIZE_TYPE pZOffset)
{

    const SIZE_TYPE index = get_global_id(0);

    if (index >= size) return;

    const VALUE_TYPE alpha = *(pAlpha + pAlphaOffset);

    pY[index + pYOffset] = operation(pZ[index + pZOffset], alpha * pX[index + pXOffset]);
}
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void axpby(const SIZE_TYPE size,
          __global VALUE_TYPE* pY,
          const SIZE_TYPE pYOffset,
          __global const VALUE_TYPE* pAlpha,
         const SIZE_TYPE pAlphaOffset,
          __global const VALUE_TYPE* pX,
          const SIZE_TYPE pXOffset,
          __global const VALUE_TYPE* pBeta,
          const SIZE_TYPE pBetaOffset,
          __global const VALUE_TYPE* pZ,
          const SIZE_TYPE pZOffset)
{

    const SIZE_TYPE index = get_global_id(0);

    if (index >= size) return;

    const VALUE_TYPE alpha = *(pAlpha + pAlphaOffset);
    const VALUE_TYPE beta = *(pBeta + pBetaOffset);

    pY[index + pYOffset] = operation(alpha * pX[index + pXOffset], beta * pZ[index + pZOffset]);
}
)"

R"(
__kernel
__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))
void scale (const SIZE_TYPE pRSize,
            __global VALUE_TYPE* pR,
            const SIZE_TYPE pROffset,
            __global const VALUE_TYPE* pY,
            const SIZE_TYPE pYOffset,
             __global const VALUE_TYPE* pAlpha,
            const SIZE_TYPE pAlphaOffset)
{
    const SIZE_TYPE i = get_global_id(0);

    if (i >= pRSize) return;

    const VALUE_TYPE alpha = *(pAlpha + pAlphaOffset);

    pR[i + pROffset] = pY[i + pYOffset]* alpha;
}
)"
