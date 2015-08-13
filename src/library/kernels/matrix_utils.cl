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

#ifndef SIZE_TYPE
#error SIZE_TYPE undefined!
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif

#ifndef WAVE_SIZE
#error WAVE_SIZE undefined!
#endif

#ifndef SUBWAVE_SIZE
#error SUBWAVE_SIZE undefined!
#endif

#if ( (SUBWAVE_SIZE > WAVE_SIZE) || (SUBWAVE_SIZE != 2 && SUBWAVE_SIZE != 4 && SUBWAVE_SIZE != 8 && SUBWAVE_SIZE != 16 && SUBWAVE_SIZE != 32 && SUBWAVE_SIZE != 64) )
# error SUBWAVE_SIZE is not  a power of two!
#endif


// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// VALUE_TYPE - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void extract_diagonal ( const SIZE_TYPE num_rows,
                     __global VALUE_TYPE * diag,
               __global const INDEX_TYPE * const csr_row_offsets,
               __global const INDEX_TYPE * const csr_col_indices,
               __global const VALUE_TYPE * const csr_values)
{

    const int global_id   = get_global_id(0);         // global workitem id
    const int local_id    = get_local_id(0);          // local workitem id
    const int thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const int vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    const int num_vectors = get_global_size(0) / SUBWAVE_SIZE;

    for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
        const int row_start = csr_row_offsets[row];
        const int row_end   = csr_row_offsets[row+1];

        for(int j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
        {
            if (csr_col_indices[j] == row)
            {
            #ifdef OP_DIAG_INVERSE
                diag[row] = (VALUE_TYPE) 1.0 / csr_values[j];
            #else
                diag[row] = csr_values[j];
            #endif
                break;
            }
        }
    }
}

)"
