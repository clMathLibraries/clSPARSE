R"(
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

#define EXTENDED_PRECISION

// Knuth's Two-Sum algorithm, which allows us to add together two floating
// point numbers and exactly tranform the answer into a sum and a
// rounding error.
// Inputs: x and y, the two inputs to be aded together.
// In/Out: *sumk_err, which is incremented (by reference) -- holds the
//         error value as a result of the 2sum calculation.
// Returns: The non-corrected sum of inputs x and y.
VALUE_TYPE two_sum( VALUE_TYPE x,
        VALUE_TYPE y,
        VALUE_TYPE *sumk_err)
{
    VALUE_TYPE sumk_s = x + y;
#ifdef EXTENDED_PRECISION
    // We use this 2Sum algorithm to perform a compensated summation,
    // which can reduce the cummulative rounding errors in our SpMV summation.
    // Our compensated sumation is based on the SumK algorithm (with K==2) from
    // Ogita, Rump, and Oishi, "Accurate Sum and Dot Product" in
    // SIAM J. on Scientific Computing 26(6) pp 1955-1988, Jun. 2005.

    // 2Sum can be done in 6 FLOPs without a branch. However, calculating
    // double precision is slower than single precision on every existing GPU.
    // As such, replacing 2Sum with Fast2Sum when using DPFP results in slightly
    // better performance. This is especially true on non-workstation GPUs with
    // low DPFP rates. Fast2Sum is faster even though we must ensure that
    // |a| > |b|. Branch divergence is better than the DPFP slowdown.
    // Thus, for DPFP, our compensated summation algorithm is actually described
    // by both Pichat and Neumaier in "Correction d'une somme en arithmetique
    // a virgule flottante" (J. Numerische Mathematik 19(5) pp. 400-406, 1972)
    // and "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
    // Summen (ZAMM Z. Angewandte Mathematik und Mechanik 54(1) pp. 39-51,
    // 1974), respectively.
    VALUE_TYPE swap;
    if (fabs(x) < fabs(y))
    {
        swap = x;
        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
    // Original 6 FLOP 2Sum algorithm.
    //VALUE_TYPE bp = sumk_s - x;
    //(*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
#endif
    return sumk_s;
}

// A method of doing the final reduction without having to copy and paste
// it a bunch of times.
// The EXTENDED_PRECISION section is done as part of the PSum2 addition,
// where we take temporary sums and errors for multiple threads and combine
// them together using the same 2Sum method.
// Inputs:  cur_sum: the input from which our sum starts
//          err: the current running cascade error for this final summation
//          partial: the local memory which holds the values to sum
//                  (we eventually use it to pass down temp. err vals as well)
//          lid: local ID of the work item calling this function.
//          thread_lane: The lane within this SUBWAVE for reduction.
//          round: This parallel summation method operates in multiple rounds
//                  to do a parallel reduction. See the blow comment for usage.
VALUE_TYPE sum2_reduce( VALUE_TYPE cur_sum,
        VALUE_TYPE *err,
        volatile __local VALUE_TYPE *partial,
        size_t lid,
        int thread_lane,
        int round)
{
    if (SUBWAVE_SIZE > round)
    {
#ifdef EXTENDED_PRECISION
        if (thread_lane < round)
            cur_sum  = two_sum(cur_sum, partial[lid + round], err);

        // We reuse the LDS entries to move the error values down into lower
        // threads. This saves LDS space, allowing higher occupancy, but requires
        // more barriers, which can reduce performance.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Have all of those upper threads pass their temporary errors
        // into a location that the lower threads can read.
        if (thread_lane >= round)
            partial[lid] = *err;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thread_lane < round) { // Add those errors in.
            *err += partial[lid + round];
            partial[lid] = cur_sum;
        }
#else
        // This is the more traditional reduction algorithm. It is up to
        // 25% faster (about 10% on average -- potentially worse on devices
        // with low double-precision calculation rates), but can result in
        // numerical inaccuracies, especially in single precision.
        cur_sum += partial[lid + round];
        partial[lid] = cur_sum;
#endif
    }
    return cur_sum;
}

// Uses macro constants:
// WAVE_SIZE  - "warp size", typically 64 (AMD) or 32 (NV)
// WG_SIZE    - workgroup ("block") size, 1D representation assumed
// INDEX_TYPE - typename for the type of integer data read by the kernel,  usually unsigned int
// VALUE_TYPE - typename for the type of floating point data, usually double
// SUBWAVE_SIZE - the length of a "sub-wave", a power of 2, i.e. 1,2,4,...,WAVE_SIZE, assigned to process a single matrix row
__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void csrmv_general (     const INDEX_TYPE num_rows,
                __global const VALUE_TYPE * const alpha,
                         const SIZE_TYPE off_alpha,
                __global const INDEX_TYPE * const row_offset,
                __global const INDEX_TYPE * const col,
                __global const VALUE_TYPE * const val,
                __global const VALUE_TYPE * const x,
                         const SIZE_TYPE off_x,
                __global const VALUE_TYPE * const beta,
                         const SIZE_TYPE off_beta,
                __global       VALUE_TYPE * y,
                         const SIZE_TYPE off_y)
{
    local volatile VALUE_TYPE sdata [WG_SIZE + SUBWAVE_SIZE / 2];

    //const int vectors_per_block = WG_SIZE/SUBWAVE_SIZE;
    const int global_id   = get_global_id(0);         // global workitem id
    const int local_id    = get_local_id(0);          // local workitem id
    const int thread_lane = local_id & (SUBWAVE_SIZE - 1);
    const int vector_id   = global_id / SUBWAVE_SIZE; // global vector id
    //const int vector_lane = local_id / SUBWAVE_SIZE;  // vector id within the workgroup
    const int num_vectors = get_global_size(0) / SUBWAVE_SIZE;

    const VALUE_TYPE _alpha = alpha[off_alpha];
    const VALUE_TYPE _beta = beta[off_beta];

    for(INDEX_TYPE row = vector_id; row < num_rows; row += num_vectors)
    {
        const int row_start = row_offset[row];
        const int row_end   = row_offset[row+1];
        VALUE_TYPE sum = 0.;

        VALUE_TYPE sumk_e = 0.;
        // It is about 5% faster to always multiply by alpha, rather than to
        // check whether alpha is 0, 1, or other and do different code paths.
        for(int j = row_start + thread_lane; j < row_end; j += SUBWAVE_SIZE)
            sum = two_sum(sum, (_alpha * val[j] * x[off_x + col[j]]), &sumk_e);
        VALUE_TYPE new_error = 0.;
        sum = two_sum(sum, sumk_e, &new_error);

        // Parallel reduction in shared memory.
        sdata[local_id] = sum;

        // This compensated summation reduces cummulative rounding errors,
        // which can become a problem on GPUs because our reduction order is
        // different than what would be used on a CPU.
        // It is based on the PSumK algorithm (with K==2) from
        // Yamanaka, Ogita, Rump, and Oishi, "A Parallel Algorithm of
        // Accurate Dot Product," in the Journal of Parallel Computing,
        // 34(6-8), pp. 392-410, Jul. 2008.
        for (int i = (WG_SIZE >> 1); i > 0; i >>= 1)
        {
            barrier( CLK_LOCAL_MEM_FENCE );
            sum = sum2_reduce(sum, &new_error, sdata, local_id, thread_lane, i);
        }

        if (thread_lane == 0)
        {
            if (_beta == 0)
                y[off_y + row] = sum + new_error;
            else
            {
                sum = two_sum(sum, _beta * y[off_y + row], &new_error);
                y[off_y + row] = sum + new_error;
            }
        }
    }
}
)"
