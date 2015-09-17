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
#if defined(ATOMIC_DOUBLE) && __OPENCL_VERSION__ < CL_VERSION_1_2
  #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #else
    #error "Double precision floating point not supported by this OpenCL implementation."
  #endif
#endif

#ifdef ATOMIC_DOUBLE
  #if defined(cl_khr_int64_base_atomics) && defined(cl_khr_int64_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
    #pragma OPENCL EXTENSION cl_khr_int64_extended_atomics : enable
  #else
    #error "Required 64-bit atomics not supported by this OpenCL implementation."
  #endif
#endif

#if defined(ATOMIC_FLOAT) || defined (ATOMIC_INT)
  #if defined(cl_khr_global_int32_base_atomics) && defined(cl_khr_global_int32_extended_atomics)
    #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : require
    #pragma OPENCL_EXTENSION cl_khr_global_int32_extended_atomics : require
  #else
    #error "Required 32-bit atomics not supported by this OpenCL implemenation."
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
        } while (atomic_cmpxchg((global unsigned ing *)ptr, prevVal, newVal) != prevVal);
#endif
}
)"

R"(
inline VALUE_TYPE operation(VALUE_TYPE A)
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

#ifdef ATOMIC_INT
    atomic_add(&pSum[0], pX[idx]);
#else
    atomic_add_float(&pSum[0], pX[idx]);
#endif

    barrier(CLK_LOCAL_MEM_FENCE);

    //at the end we want to modify the final value like in L2 norm.
    pSum[0] = operation(pSum[0]);
}
)"
