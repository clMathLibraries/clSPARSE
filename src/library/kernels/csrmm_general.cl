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

#if defined DOUBLE
    #ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
    #elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
    #else
    #error "Double precision floating point not supported by OpenCL implementation."
    #endif
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
)"

R"(
void
csrmv( const INDEX_TYPE num_rows,
        global const VALUE_TYPE * const alpha,
        const SIZE_TYPE off_alpha,
        global const INDEX_TYPE * const restrict row_offset,
        global const INDEX_TYPE * const restrict col,
        global const VALUE_TYPE * const restrict val,
        global const VALUE_TYPE * restrict x,
        const SIZE_TYPE ldx,
        const SIZE_TYPE off_x,
        global const VALUE_TYPE * const beta,
        const SIZE_TYPE off_beta,
        global VALUE_TYPE * restrict y,
        const SIZE_TYPE ldy,
        const SIZE_TYPE off_y,
        local VALUE_TYPE* sdata
        )
{
    //const int vectors_per_block = WG_SIZE/SUBWAVE_SIZE;
    const int global_id = get_global_id( 0 );         // global workitem id
    const int local_id = get_local_id( 0 );          // local workitem id
    const int thread_lane = local_id & ( SUBWAVE_SIZE - 1 );
    const int vector_id = global_id / SUBWAVE_SIZE; // global vector id
    //const int vector_lane = local_id / SUBWAVE_SIZE;  // vector id within the workgroup
    const int num_vectors = get_global_size( 0 ) / SUBWAVE_SIZE;

    const VALUE_TYPE _alpha = alpha[ off_alpha ];
    const VALUE_TYPE _beta = beta[ off_beta ];

    for( INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors )
    {
        const int row_start = row_offset[ row ];
        const int row_end = row_offset[ row + 1 ];
        VALUE_TYPE sum = (VALUE_TYPE)0;

        for( int j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE )
        {
            if( _alpha == 1 )
                sum = fma( val[ j ], x[ off_x + ( col[ j ] * ldx ) ], sum );
            else if( _alpha == 0 )
                sum = 0;
            else
                sum = fma( _alpha * val[ j ], x[ off_x + ( col[ j ] * ldx ) ], sum );
        }

        //parllel reduction in shared memory
        sdata[ local_id ] = sum;
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 32 ) sdata[ local_id ] = sum += sdata[ local_id + 32 ];
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 16 ) sdata[ local_id ] = sum += sdata[ local_id + 16 ];
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 8 )  sdata[ local_id ] = sum += sdata[ local_id + 8 ];
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 4 )  sdata[ local_id ] = sum += sdata[ local_id + 4 ];
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 2 )  sdata[ local_id ] = sum += sdata[ local_id + 2 ];
        barrier( CLK_LOCAL_MEM_FENCE );
        if( SUBWAVE_SIZE > 1 )                    sum += sdata[ local_id + 1 ];

        if( thread_lane == 0 )
        {
            if( _beta == 1 )
                y[ off_y + ( row * ldy ) ] = sum + y[ off_y + ( row * ldy ) ];
            else if( _beta == 0 )
                y[ off_y + ( row * ldy ) ] = sum;
            else
                y[ off_y + ( row * ldy ) ] = sum + _beta * y[ off_y + ( row * ldy ) ];
        }
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
kernel
__attribute__( ( reqd_work_group_size( WG_SIZE, 1, 1 ) ) )
void csrmv_batched( const INDEX_TYPE num_rows,
            global const VALUE_TYPE * const alpha,
            const SIZE_TYPE off_alpha,
            global const INDEX_TYPE * const restrict row_offset,
            global const INDEX_TYPE * const restrict col,
            global const VALUE_TYPE * const restrict val,
            global const VALUE_TYPE * const restrict denseB,
            const SIZE_TYPE ldB,
            const SIZE_TYPE off_B,
            global const VALUE_TYPE * const beta,
            const SIZE_TYPE off_beta,
            global VALUE_TYPE * restrict denseC,
            const SIZE_TYPE num_rows_C,
            const SIZE_TYPE num_cols_C,
            const SIZE_TYPE ldC,
            const SIZE_TYPE off_C )
{
    local VALUE_TYPE sdata[ WG_SIZE + SUBWAVE_SIZE / 2 ];

    //  The current implementation of csrmm is implemented as a batched csrmv
    //  The loop iterates on the number of columns in the output matrix, and we increment
    //  the global pointers to the dense B and C matrices a column for each iteration.
    for( SIZE_TYPE curr_col = 0; curr_col < num_cols_C; ++curr_col )
    {
        csrmv( num_rows, alpha, off_alpha, row_offset, col, val, denseB + curr_col, ldB, off_B, beta, off_beta, denseC + curr_col, ldC, off_C, sdata );
    }
}
)"
