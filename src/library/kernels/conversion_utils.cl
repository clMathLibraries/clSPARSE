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

#ifndef INDEX_TYPE
#error "INDEX_TYPE undefined!"
#endif

#ifndef WG_SIZE
#error "WG_SIZE undefined!"
#endif

#ifndef WAVE_SIZE
#define WAVE_SIZE 32
#endif

#ifndef SUBWAVE_SIZE
#error "SUBWAVE_SIZE undefined!"
#endif

#if ( (SUBWAVE_SIZE > WAVE_SIZE) || (SUBWAVE_SIZE != 2 && SUBWAVE_SIZE != 4 && SUBWAVE_SIZE != 8 && SUBWAVE_SIZE != 16 && SUBWAVE_SIZE != 32 && SUBWAVE_SIZE != 64) )
#error "SUBWAVE_SIZE is not  a power of two!"
#endif

)"

R"(
// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// VALUE_TYPE - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void offsets_to_indices ( const SIZE_TYPE          num_rows,
                __global const  INDEX_TYPE * const offsets,
                __global        INDEX_TYPE *       indices)
{

    const SIZE_TYPE global_id   = get_global_id(0);         // global workitem id
    const SIZE_TYPE local_id    = get_local_id(0);          // local workitem id
    const SIZE_TYPE thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const SIZE_TYPE vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    const SIZE_TYPE num_vectors = get_global_size(0) / SUBWAVE_SIZE;

    for(SIZE_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
        const INDEX_TYPE row_start = offsets[row];
        const INDEX_TYPE row_end   = offsets[row+1];

        for(INDEX_TYPE j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
            indices[j] = row;
    }
}

)"

R"(
// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// VALUE_TYPE - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void transform_csr_to_dense(
                         const SIZE_TYPE num_rows,
                         const SIZE_TYPE num_cols,
                __global const INDEX_TYPE * const row_offset,
                __global const INDEX_TYPE * const col,
                __global const VALUE_TYPE * const val,
                __global       VALUE_TYPE * A)
{

    const SIZE_TYPE global_id   = get_global_id(0);         // global workitem id
    const SIZE_TYPE local_id    = get_local_id(0);          // local workitem id
    const SIZE_TYPE thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const SIZE_TYPE vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    const SIZE_TYPE num_vectors = get_global_size(0) / SUBWAVE_SIZE;

    for(SIZE_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
        const INDEX_TYPE row_start = row_offset[row];
        const INDEX_TYPE row_end   = row_offset[row+1];

        for(INDEX_TYPE j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
            A[row * num_cols + col[j]] = val[j];
    }
}
)"


R"(
//Kerne fills the locations buffer in the positions of non-zero values in dense matrix,
//TODO:: would it be better if the locations will be filled with zeros and then filled with ones only?
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void scan_nonzero_locations( SIZE_TYPE size,
                     __global const VALUE_TYPE * const A,
                     __global INDEX_TYPE* locations)
{
    SIZE_TYPE index = get_global_id(0);

    if (index < size)
    {
        if (A[index] != 0)
            locations[index] = 1;
        else
            locations[index] = 0;
    }
}
)"

R"(
// according to locations arrangement fill the values of csr matrix
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void scatter_coo_locations(SIZE_TYPE num_rows,
                           SIZE_TYPE num_cols,
                           SIZE_TYPE size,
                           __global VALUE_TYPE* Avals,
                           __global INDEX_TYPE* nnz_locations,
                           __global INDEX_TYPE* coo_indexes,
                           __global INDEX_TYPE* row_indices,
                           __global INDEX_TYPE* col_indices,
                           __global VALUE_TYPE* values)
{
    SIZE_TYPE index = get_global_id(0);

    if (nnz_locations[index] == 1 && index < size)
    {
        INDEX_TYPE row_index = index / num_cols;
        INDEX_TYPE col_index = index % num_cols; //modulo taks many cycles to calculate

        INDEX_TYPE location = coo_indexes[index];
        row_indices[ location ] = row_index;
        col_indices[ location ] = col_index;

        values [ location ] = Avals [index];
    }
}

)"
