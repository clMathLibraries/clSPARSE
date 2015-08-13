R"(
/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef INDEX_TYPE
#error INDEX_TYPE undefined!
#endif

#ifndef VALUE_TYPE
#error VALUE_TYPE undefined!
#endif

__kernel
void process_scaninput ( int total,
			__global  VALUE_TYPE *A,
			__global  int *scan_input )
{

    int tid   = get_global_id(0);
    if (tid >= total)
        return;

    if (A[tid] != 0)
       scan_input[tid] = 1;
    else
       scan_input[tid] = 0;

}

__kernel
void spread_value( int m, int n, int total,
                  __global VALUE_TYPE *A,
                  __global int *scan_input,
                  __global int *scan_output,
                  __global int *row,
                  __global int *col,
                  __global VALUE_TYPE*val){

	int tid   = get_global_id(0);
	if (scan_input[tid] == 1 && tid < total){
	    int x_idx = tid / n;
            int y_idx = tid % n;
            row[scan_output[tid]] = x_idx;
            col[scan_output[tid]] = y_idx;
            val[scan_output[tid]] = A[tid];
	}
}

)"
