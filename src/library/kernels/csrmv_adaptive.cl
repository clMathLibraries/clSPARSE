R"(
#ifdef DOUBLE
  #define FPTYPE double

  #ifdef cl_khr_fp64
      #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #elif defined(cl_amd_fp64)
      #pragma OPENCL EXTENSION cl_amd_fp64 : enable
  #else
      #error "Double precision floating point not supported by OpenCL implementation."
  #endif
#else
  #define FPTYPE float
#endif

#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics: enable

FPTYPE atomic_add_float_extended( global FPTYPE * restrict const ptr,
                                  const FPTYPE temp,
                                  FPTYPE * restrict const old_sum )
{
#ifdef DOUBLE
	unsigned long newVal;
	unsigned long prevVal;
	do
	{
		prevVal = as_ulong(*ptr);
		newVal = as_ulong(temp + *ptr);
	} while (atom_cmpxchg((global unsigned long *)ptr, prevVal, newVal) != prevVal);
    if (old_sum != 0)
        *old_sum = as_double(prevVal);
    return as_double(newVal);
#else
	unsigned int newVal;
	unsigned int prevVal;
	do
	{
		prevVal = as_uint(*ptr);
		newVal = as_uint(temp + *ptr);
	} while (atomic_cmpxchg((global unsigned int *)ptr, prevVal, newVal) != prevVal);
    if (old_sum != 0)
        *old_sum = as_float(prevVal);
    return as_float(newVal);
#endif
}

void atomic_add_float( global void * const ptr, const FPTYPE temp )
{
    atomic_add_float_extended(ptr, temp, 0);
}

// Knuth's Two-Sum algorithm, which allows us to add together two floating
// point numbers and exactly tranform the answer into a sum and a
// rounding error.
// Inputs: x and y, the two inputs to be aded together.
// In/Out: *sumk_err, which is incremented (by reference) -- holds the
//         error value as a result of the 2sum calculation.
// Returns: The non-corrected sum of inputs x and y.
FPTYPE two_sum( FPTYPE x,
                FPTYPE y,
                FPTYPE * restrict const sumk_err )
{
    const FPTYPE sumk_s = x + y;
#ifdef EXTENDED_PRECISION
    // We use this 2Sum algorithm to perform a compensated summation,
    // which can reduce the cummulative rounding errors in our SpMV summation.
    // Our compensated sumation is based on the SumK algorithm (with K==2) from
    // Ogita, Rump, and Oishi, "Accurate Sum and Dot Product" in
    // SIAM J. on Scientific Computing 26(6) pp 1955-1988, Jun. 2005.

    // 2Sum can be done in 6 FLOPs without a branch. However, calculating
    // double precision is slower than single precision on every existing GPU.
    // As such, replacing 2Sum with Fast2Sum when using DPFP results in better
    // performance (~5% higher total). This is true even though we must ensure
    // that |a| > |b|. Branch divergence is better than the DPFP slowdown.
    // Thus, for DPFP, our compensated summation algorithm is actually described
    // by both Pichat and Neumaier in "Correction d'une somme en arithmetique
    // a virgule flottante" (J. Numerische Mathematik 19(5) pp. 400-406, 1972)
    // and "Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
    // Summen (ZAMM Z. Angewandte Mathematik und Mechanik 54(1) pp. 39-51,
    // 1974), respectively.
    if (fabs(x) < fabs(y))
    {
        const FPTYPE swap = x;
        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
    // Original 6 FLOP 2Sum algorithm.
    //const FPTYPE bp = sumk_s - x;
    //(*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
#endif
    return sumk_s;
}

FPTYPE atomic_two_sum_float( global FPTYPE * restrict const x_ptr,
                            FPTYPE y,
                            FPTYPE * restrict const sumk_err )
{
    // Have to wait until the return from the atomic op to know what X was.
    FPTYPE sumk_s = 0.;
#ifdef EXTENDED_PRECISION
    FPTYPE x;
    sumk_s = atomic_add_float_extended(x_ptr, y, &x);
    if (fabs(x) < fabs(y))
    {
        const FPTYPE swap = x;
        x = y;
        y = swap;
    }
    (*sumk_err) += (y - (sumk_s - x));
#else
    atomic_add_float(x_ptr, y);
#endif
    return sumk_s;
}

// A method of doing the final reduction in CSR-Vector without having to copy
// and paste it a bunch of times.
// The EXTENDED_PRECISION section is done as part of a compensated summation
// meant to reduce cummulative rounding errors. This can become a problem on
// GPUs because the reduction order is different than what would be used on a
// CPU. It is based on the PSumK algorithm (with K==2) from Yamanaka, Ogita,
// Rump, and Oishi, "A Parallel Algorithm of Accurate Dot Product," in
// J. Parallel Computing, 34(6-8), pp. 392-410, Jul. 2008.
// A version of this method is also used in csrmv_general.
// Inputs:  cur_sum: the input from which our sum starts
//          err: the current running cascade error for this final summation
//          partial: the local memory which holds the values to sum
//                  (we eventually use it to pass down temp. err vals as well)
//          lid: local ID of the work item calling this function.
//          thread_lane: The lane within this thread's row.
//          max_size: This parallel summation method operates in multiple rounds
//                  to do a parallel reduction. This is the length of each row.
//          reduc_size: As you reduce data down, this tells you how far away
//                 you will grab a value and add to your own local sum value.
FPTYPE sum2_reduce( FPTYPE cur_sum,
        FPTYPE * restrict const err,
        __local FPTYPE * restrict const partial,
        const unsigned int lid,
        const unsigned int thread_lane,
        const unsigned int max_size,
        const unsigned int reduc_size )
{
    if ( max_size > reduc_size )
    {
#ifdef EXTENDED_PRECISION
        if (thread_lane < reduc_size)
            cur_sum  = two_sum(cur_sum, partial[lid + reduc_size], err);

        // We reuse the LDS entries to move the error values down into lower
        // threads. This saves LDS space, allowing higher occupancy, but requires
        // more barriers, which can reduce performance.
        barrier(CLK_LOCAL_MEM_FENCE);
        // Have all of those upper threads pass their temporary errors
        // into a location that the lower threads can read.
        if (thread_lane >= reduc_size)
            partial[lid] = *err;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (thread_lane < reduc_size) // Add those errors in.
        {
            *err += partial[lid + reduc_size];
            partial[lid] = cur_sum;
        }
#else
        cur_sum += partial[lid + reduc_size];
        partial[lid] = cur_sum;
#endif
    }
    return cur_sum;
}

__kernel void
csrmv_adaptive(__global const FPTYPE * restrict const vals,
                       __global const unsigned int * restrict const cols,
                       __global const unsigned int * restrict const rowPtrs,
                       __global const FPTYPE * restrict const vec,
                       __global FPTYPE * restrict const out,
                       __global unsigned long * restrict const rowBlocks,
                       __global const FPTYPE * restrict const pAlpha,
                       __global const FPTYPE * restrict const pBeta)
{
   __local FPTYPE partialSums[BLOCKSIZE];
   const unsigned int gid = get_group_id(0);
   const unsigned int lid = get_local_id(0);
   const FPTYPE alpha = *pAlpha;
   const FPTYPE beta = *pBeta;

   // The row blocks buffer holds a packed set of information used to inform each
   // workgroup about how to do its work:
   //
   // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
   // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
   // |------------Row Information------------|--------^|---WG ID within a long row---|
   // |                                       |    flag/|or # reduce threads for short|
   //
   // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
   // row it will be working on. When one workgroup calculates multiple rows, this
   // rowBlock entry and the next one tell it the range of rows to work on.
   // The lower 24 bits are used whenever multiple workgroups calculate a single long
   // row. This tells each workgroup its ID within that row, so it knows which
   // part of the row to operate on.
   // Alternately, on "short" row blocks, the lower bits are used to communicate
   // the number of threads that should be used for the reduction. Pre-calculating
   // this on the CPU-side results in a noticable performance uplift on many matrices.
   // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
   // know when the first workgroup for that row has finished initializing the output
   // value. While this bit is the same as the first workgroup's flag bit, this
   // workgroup will spin-loop.
   unsigned int row = ((rowBlocks[gid] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));
   const unsigned int stop_row = ((rowBlocks[gid + 1] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));
   const unsigned int num_rows = stop_row - row;

   // Get the "workgroup within this long row" ID out of the bottom bits of the row block.
   const unsigned int wg = rowBlocks[gid] & ((1 << WGBITS) - 1);

   // Any workgroup only calculates, at most, BLOCK_MULTIPLIER*BLOCKSIZE items in a row.
   // If there are more items in this row, we assign more workgroups.
   unsigned int vecStart = rowPtrs[row] + (wg * BLOCK_MULTIPLIER*BLOCKSIZE);
   unsigned int vecEnd = (rowPtrs[row + 1] > vecStart + BLOCK_MULTIPLIER*BLOCKSIZE) ? vecStart + BLOCK_MULTIPLIER*BLOCKSIZE : rowPtrs[row + 1];

   FPTYPE temp_sum = 0.;
   FPTYPE sumk_e = 0.;
   FPTYPE new_error = 0.;

   // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
   // If this is zero (long rows) or one (final workgroup in a long row, or a single
   // row in a row block), we want to use the CSR-Vector algorithm(s).
   // We have found, through experimentation, that CSR-Vector is generally faster
   // when working on 2 rows, due to its simplicity and better reduction method.
   if (num_rows > ROWS_FOR_VECTOR)
   {
       // CSR-Stream case. See Sections III.A and III.B in the SC'14 paper:
       // "Efficient Sparse Matrix-Vector Multiplication on GPUs using the CSR Storage Format"
       // for a detailed description of CSR-Stream.
       // In a nutshell, the idea is to use all of the threads to stream the matrix
       // values into the local memory in a fast, coalesced manner. After that, the
       // per-row reductions are done out of the local memory, which is designed
       // to handle non-coalsced accesses.

       // The best method for reducing the local memory values depends on the number
       // of rows. The SC'14 paper discusses a "CSR-Scalar" style reduction where
       // each thread reduces its own row. This yields good performance if there
       // are many (relatively short) rows. However, if they are few (relatively
       // long) rows, it's actually better to perform a tree-style reduction where
       // multiple threads team up to reduce the same row.

       // The calculation below tells you how many threads this workgroup can allocate
       // to each row, assuming that every row gets the same number of threads.
       // We want the closest lower (or equal) power-of-2 to this number --
       // that is how many threads can work in each row's reduction using our algorithm.
       // For instance, with workgroup size 256, 2 rows = 128 threads, 3 rows = 64
       // threads, 4 rows = 64 threads, 5 rows = 32 threads, etc.
       //int numThreadsForRed = get_local_size(0) >> ((CHAR_BIT*sizeof(unsigned int))-clz(num_rows-1));
       const unsigned int numThreadsForRed = wg; // Same calculation as above, done on host.

       // Stream all of this row block's matrix values into local memory.
       // Perform the matvec in parallel with this work.
       const unsigned int col = rowPtrs[row] + lid;
      if (gid != (get_num_groups(0) - 1))
      {
          for(int i = 0; i < BLOCKSIZE; i += WGSIZE)
              partialSums[lid + i] = alpha * vals[col + i] * vec[cols[col + i]];
      }
      else
      {
          // This is required so that we stay in bounds for vals[] and cols[].
          // Otherwise, if the matrix's endpoints don't line up with BLOCKSIZE,
          // we will buffer overflow. On today's dGPUs, this doesn't cause problems.
          // The values are within a dGPU's page, which is zeroed out on allocation.
          // However, this may change in the future (e.g. with shared virtual memory.)
          // This causes a minor performance loss because this is the last workgroup
          // to be launched, and this loop can't be unrolled.
          const unsigned int max_to_load = rowPtrs[stop_row] - rowPtrs[row];
          for(int i = 0; i < max_to_load; i += WGSIZE)
              partialSums[lid + i] = alpha * vals[col + i] * vec[cols[col + i]];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if(numThreadsForRed > 1)
      {
          // In this case, we want to have the workgroup perform a tree-style reduction
          // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
          // a row into {numThreadsForRed} locations in local memory.
          // After that, the entire workgroup does a parallel reduction, and each
          // row ends up with an individual answer.

          // {numThreadsForRed} adjacent threads all work on the same row, so their
          // start and end values are the same.
          // numThreadsForRed guaranteed to be a power of two, so the clz code below
          // avoids an integer divide. ~2% perf gain in EXTRA_PRECISION.
          //size_t st = lid/numThreadsForRed;
          const unsigned int local_row = row + (lid >> (31 - clz(numThreadsForRed)));
          const unsigned int local_first_val = rowPtrs[local_row] - rowPtrs[row];
          const unsigned int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
          const unsigned int threadInBlock = lid & (numThreadsForRed - 1);

          // Not all row blocks are full -- they may have an odd number of rows. As such,
          // we need to ensure that adjacent-groups only work on real data for this rowBlock.
          if(local_row < stop_row)
          {
              // only works when numThreadsForRed is a power of 2
              for(int i = 0; i < workForEachThread; i++)
                  temp_sum = two_sum(temp_sum, partialSums[local_first_val + i*numThreadsForRed + threadInBlock], &sumk_e);

              // The last few values (the remainder of this row) also need to be aded in.
              int local_cur_val = local_first_val + numThreadsForRed*workForEachThread;
              if(threadInBlock < local_last_val - local_cur_val)
                  temp_sum = two_sum(temp_sum, partialSums[local_cur_val + threadInBlock], &sumk_e);
          }
          barrier(CLK_LOCAL_MEM_FENCE);

          temp_sum = two_sum(temp_sum, sumk_e, &new_error);
          partialSums[lid] = temp_sum;

          // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
          // values sitting in the local memory. This means that, roughly, the beginning of
          // LDS is full up to {workgroup size} entries.
          // Now we perform a parallel reduction that sums together the answers for each
          // row in parallel, leaving us an answer in 'temp_sum' for each row.
          for (int i = (WGSIZE >> 1); i > 0; i >>= 1)
          {
              barrier( CLK_LOCAL_MEM_FENCE );
              temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, lid, threadInBlock, numThreadsForRed, i);
          }

          if (threadInBlock == 0 && local_row < stop_row)
          {
              // All of our write-outs check to see if the output vector should first be zeroed.
              // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
              // performance improvement.
              if (beta != 0.)
                  temp_sum = two_sum(beta * out[local_row], temp_sum, &new_error);
              out[local_row] = temp_sum + new_error;
          }
      }
      else
      {
          // In this case, we want to have each thread perform the reduction for a single row.
          // Essentially, this looks like performing CSR-Scalar, except it is computed out of local memory.
          // However, this reduction is also much faster than CSR-Scalar, because local memory
          // is designed for scatter-gather operations.
          // We need a while loop because there may be more rows than threads in the WG.
          unsigned int local_row = row + lid;
          while(local_row < stop_row)
          {
              int local_first_val = (rowPtrs[local_row] - rowPtrs[row]);
              int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
              temp_sum = 0.;
              sumk_e = 0.;
              for (int local_cur_val = local_first_val; local_cur_val < local_last_val; local_cur_val++)
                  temp_sum = two_sum(temp_sum, partialSums[local_cur_val], &sumk_e);

              // After you've done the reduction into the temp_sum register,
              // put that into the output for each row.
              if (beta != 0.)
                  temp_sum = two_sum(beta * out[local_row], temp_sum, &sumk_e);
              out[local_row] = temp_sum + sumk_e;
              local_row += WGSIZE;
          }
      }
   }
   else if (num_rows >= 1 && !wg) // CSR-Vector case.
   {
       // ^^ The above check says that if this workgroup is supposed to work on <= ROWS_VECTOR
       // number of rows then we should do the CSR-Vector algorithm. If we want this row to be
       // done with CSR-LongRows, then all of its workgroups (except the last one) will have the
       // same stop_row and row. The final workgroup in a LongRow will have stop_row and row
       // different, but the internal "wg" number will be non-zero.

       // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
       // numbers of rows), then it needs to iterate until it reaches the stop_row.
       // We don't check <= stop_row because of the potential for unsigned overflow.
       while (row < stop_row)
       {
           // Any workgroup only calculates, at most, BLOCKSIZE items in this row.
           // If there are more items in this row, we use CSR-LongRows.
           temp_sum = 0.;
           sumk_e = 0.;
           new_error = 0.;
           vecStart = rowPtrs[row];
           vecEnd = rowPtrs[row+1];

           // Load in a bunch of partial results into your register space, rather than LDS (no contention)
           // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
           // Using a long induction variable to make sure unsigned int overflow doesn't break things.
           for (long j = vecStart + lid; j < vecEnd; j+=WGSIZE)
           {
               const unsigned int col = cols[(unsigned int)j];
               temp_sum = two_sum(temp_sum, alpha * vals[(unsigned int)j] * vec[col], &sumk_e);
           }

           temp_sum = two_sum(temp_sum, sumk_e, &new_error);
           partialSums[lid] = temp_sum;

           // Reduce partial sums
           for (int i = (WGSIZE >> 1); i > 0; i >>= 1)
           {
               barrier( CLK_LOCAL_MEM_FENCE);
               temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, lid, lid, WGSIZE, i);
           }

           if (lid == 0UL)
           {
               if (beta != 0.)
                   temp_sum = two_sum(beta * out[row], temp_sum, &new_error);
               out[row] = temp_sum + new_error;
           }
           row++;
       }
   }
   else
   {
       // In CSR-LongRows, we have more than one workgroup calculating this row.
       // The output values for those types of rows are stored using atomic_add, because
       // more than one parallel workgroup's value makes up the final answer.
       // Unfortunately, this makes it difficult to do y=Ax, rather than y=Ax+y, because
       // the values still left in y will be added in using the atomic_add.
       //
       // Our solution is to have the first workgroup in one of these long-rows cases
       // properly initaizlie the output vector. All the other workgroups working on this
       // row will spin-loop until that workgroup finishes its work.

       // First, figure out which workgroup you are in the row. Bottom 24 bits.
       // You can use that to find the global ID for the first workgroup calculating
       // this long row.
       const unsigned int first_wg_in_row = gid - (rowBlocks[gid] & ((1UL << WGBITS) - 1UL));
       const unsigned int compare_value = rowBlocks[gid] & (1UL << WGBITS);

       // Bit 24 in the first workgroup is the flag that everyone waits on.
       if(gid == first_wg_in_row && lid == 0UL)
       {
           // The first workgroup handles the output initialization.
           if (beta != 0.)
               out[row] *= beta;
           else
               out[row] = 0.;
#ifdef EXTENDED_PRECISION
           rowBlocks[get_num_groups(0) + gid + 1] = 0UL;
#endif
           atom_xor(&rowBlocks[first_wg_in_row], (1UL << WGBITS)); // Release other workgroups.
       }
       // For every other workgroup, bit 24 holds the value they wait on.
       // If your bit 24 == first_wg's bit 24, you spin loop.
       // The first workgroup will eventually flip this bit, and you can move forward.
       barrier(CLK_GLOBAL_MEM_FENCE);
       while(gid != first_wg_in_row &&
               lid == 0UL &&
               ((atom_max(&rowBlocks[first_wg_in_row], 0UL) & (1UL << WGBITS)) == compare_value));
       barrier(CLK_GLOBAL_MEM_FENCE);

       // After you've passed the barrier, update your local flag to make sure that
       // the next time through, you know what to wait on.
       if (gid != first_wg_in_row && lid == 0UL)
           rowBlocks[gid] ^= (1UL << WGBITS);

       // All but the final workgroup in a long-row collaboration have the same start_row
       // and stop_row. They only run for one iteration.
       // Load in a bunch of partial results into your register space, rather than LDS (no contention)
       // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
       const unsigned int col = vecStart + lid;
       if (row == stop_row) // inner thread, we can hardcode/unroll this loop
       {
           // Don't put BLOCK_MULTIPLIER*BLOCKSIZE as the stop point, because
           // some GPU compilers will *aggressively* unroll this loop.
           // That increases register pressure and reduces occupancy.
           for (int j = 0; j < (int)(vecEnd - col); j += WGSIZE)
           {
               temp_sum = two_sum(temp_sum, alpha * vals[col + j] * vec[cols[col + j]], &sumk_e);
#if 2*WGSIZE <= BLOCK_MULTIPLIER*BLOCKSIZE
               // If you can, unroll this loop once. It somewhat helps performance.
               j += WGSIZE;
               temp_sum = two_sum(temp_sum, alpha * vals[col + j] * vec[cols[col + j]], &sumk_e);
#endif
           }
       }
       else
       {
           for(int j = 0; j < (int)(vecEnd - col); j += WGSIZE)
               temp_sum = two_sum(temp_sum, alpha * vals[col + j] * vec[cols[col + j]], &sumk_e);
       }

       temp_sum = two_sum(temp_sum, sumk_e, &new_error);
       partialSums[lid] = temp_sum;

       // Reduce partial sums
       for (int i = (WGSIZE >> 1); i > 0; i >>= 1)
       {
           barrier( CLK_LOCAL_MEM_FENCE);
           temp_sum = sum2_reduce(temp_sum, &new_error, partialSums, lid, lid, WGSIZE, i);
       }

       if (lid == 0UL)
       {
           atomic_two_sum_float(&out[row], temp_sum, &new_error);

#ifdef EXTENDED_PRECISION
           // The last half of the rowBlocks buffer is used to hold errors.
           atomic_add_float(&(rowBlocks[get_num_groups(0)+first_wg_in_row+1]), new_error);
           // Coordinate across all of the workgroups in this coop in order to have
           // the last workgroup fix up the error values.
           // If this workgroup's row is different than the next workgroup's row
           // then this is the last workgroup -- it's this workgroup's job to add
           // the error values into the final sum.
           if (row != stop_row)
           {
               // Go forward once your ID is the same as the low order bits of the
               // coop's first workgroup. That value will be used to store the number
               // of threads that have completed so far. Once all the previous threads
               // are done, it's time to send out the errors!
               while((atom_max(&rowBlocks[first_wg_in_row], 0UL) & ((1UL << WGBITS) - 1)) != wg);

#ifdef DOUBLE
               new_error = as_double(rowBlocks[get_num_groups(0)+first_wg_in_row+1]);
#else
               new_error = as_float((int)rowBlocks[get_num_groups(0)+first_wg_in_row+1]);
#endif
               // Don't need to work atomically here, because this is the only workgroup
               // left working on this row.
               out[row] += new_error;
               rowBlocks[get_num_groups(0)+first_wg_in_row+1] = 0UL;

               // Reset the rowBlocks low order bits for next time.
               rowBlocks[first_wg_in_row] = rowBlocks[gid] - wg;
           }
           else
           {
               // Otherwise, increment the low order bits of the first thread in this
               // coop. We're using this to tell how many workgroups in a coop are done.
               // Do this with an atomic, since other threads may be doing this too.
               const unsigned long no_warn = atom_inc(&rowBlocks[first_wg_in_row]);
           }
#endif
       }
   }
}
)"
