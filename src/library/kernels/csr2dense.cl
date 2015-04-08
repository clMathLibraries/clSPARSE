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
#define WAVE_SIZE 64
#endif

#ifndef SUBWAVE_SIZE
#define SUBWAVE_SIZE WAVE_SIZE
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
void csr2dense ( const INDEX_TYPE num_rows,
                         const INDEX_TYPE num_cols,
                __global const INDEX_TYPE * const row_offset,
                __global const INDEX_TYPE * const col,
                __global const VALUE_TYPE * const val,
                __global       VALUE_TYPE * A)
{

    const int global_id   = get_global_id(0);         // global workitem id
    const int local_id    = get_local_id(0);          // local workitem id
    const int thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const int vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    const int num_vectors = get_global_size(0) / SUBWAVE_SIZE;

    for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
        const int row_start = row_offset[row];
        const int row_end   = row_offset[row+1];

        for(int j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
            A[row * num_cols + col[j]] = val[j];
    }
}
