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

#define WGSIZE 256

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

int lowerPowerOf2(int num)
{
        num --;

        num |= num >> 1;
        num |= num >> 2;
        num |= num >> 4;
        num |= num >> 8;
        num |= num >> 16;

        num ++;

        num >>= 1;

        return num;
}

void atomic_add_float (global FPTYPE *ptr, FPTYPE temp)
{
#ifdef DOUBLE
	unsigned long newVal;
	unsigned long prevVal;
	do
	{
		prevVal = as_ulong(*ptr);
		newVal = as_ulong(temp + *ptr);
	} while (atom_cmpxchg((global unsigned long *)ptr, prevVal, newVal) != prevVal);

#else
	unsigned int newVal;
	unsigned int prevVal;
	do
	{
		prevVal = as_uint(*ptr);
		newVal = as_uint(temp + *ptr);
	} while (atomic_cmpxchg((global unsigned int *)ptr, prevVal, newVal) != prevVal);
#endif
}

__kernel void
csrmv_adaptive(__global const FPTYPE * restrict vals,
                       __global const int * restrict cols,
                       __global const int * restrict rowPtrs,
                       __global const FPTYPE * restrict vec,
                       __global FPTYPE * restrict out,
                       __global unsigned long * restrict rowBlocks,
                       __global FPTYPE * pAlpha,
                       __global FPTYPE * pBeta)
{
   __local FPTYPE partialSums[BLOCKSIZE];
   size_t gid = get_group_id(0);
   size_t lid = get_local_id(0);
   const FPTYPE alpha = *pAlpha;
   const FPTYPE beta = *pBeta;

   // The row blocks buffer holds a packed set of information used to inform each
   // workgroup about how to do its work:
   //
   // |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
   // |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
   // |------------Row Information------------|----flag^|---WG ID within a long row---|
   //
   // The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
   // row it will be working on. When one workgroup calculates multiple rows, this
   // rowBlock entry and the next one tell it the range of rows to work on.
   // The lower 24 bits are used whenever multiple workgroups calculate a single long
   // row. This tells each workgroup its ID within that row, so it knows which
   // part of the row to operate on.
   // Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
   // know when the first workgroup for that row has finished initializing the output
   // value. While this bit is the same as the first workgroup's flag bit, this
   // workgroup will spin-loop.
   unsigned int row = ((rowBlocks[gid] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));
   unsigned int stop_row = ((rowBlocks[gid + 1] >> (64-ROWBITS)) & ((1UL << ROWBITS) - 1UL));

   // If the next row block starts more than 2 rows away, then we choose CSR-Stream.
   // If this is zero (long rows) or one (final workgroup in a long row, or a single
   // row in a row block), we want to use the CSR-Vector algorithm.
   // We have found, through experimentation, that CSR-Vector is generally faster
   // when working on 2 rows, due to its simplicity and better reduction method.
   if (stop_row - row > 2)
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

       // The calculations below tell you how many threads this workgroup can allocate
       // to each row, assuming that every row gets the same number of threads.
       // We want the closest lower-power-of-2 to this number -- that is how many
       // threads can work in each row's reduction using our algorithm.
      int possibleThreadsRed = get_local_size(0)/(stop_row - row);
      int numThreadsForRed = lowerPowerOf2(possibleThreadsRed);

      unsigned int local_row = row + lid;

       // Stream all of this row block's matrix values into local memory.
       // Perform the matvec in parallel with this work.
      int col = rowPtrs[row] + lid;
      if (gid != (get_num_groups(0) - 1))
      {
          for(int i = 0; i < BLOCKSIZE; i += 256)
          {
              partialSums[lid + i] = vals[col + i] * vec[cols[col + i]];
          }
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
          int max_to_load = rowPtrs[stop_row] - rowPtrs[row];
          for(int i = 0; i < max_to_load; i += 256)
          {
              partialSums[lid + i] = vals[col + i] * vec[cols[col + i]];
          }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      if(numThreadsForRed > 1)
      {
          // In this case, we want to have the workgroup perform a tree-style reduction
          // of each row. {numThreadsForRed} adjacent threads team up to linearly reduce
          // a row into {numThreadsForRed} locations in local memory.
          // After that, a single thread from each of those teams linearly walks through
          // the local memory values for that row and reduces to the final output value.
         FPTYPE temp = 0.;

         // {numThreadsForRed} adjacent threads all work on the same row, so their
         // start and end values are the same.
         size_t st = lid/numThreadsForRed;
         int local_first_val = (rowPtrs[row + st] - rowPtrs[row]);
         int local_last_val = rowPtrs[row + st + 1] - rowPtrs[row];
         int workForEachThread = (local_last_val - local_first_val)/numThreadsForRed;
         size_t threadInBlock = lid & (numThreadsForRed - 1);

         // Not all row blocks are full -- they may have an odd number of rows. As such,
         // we need to ensure that adjacent-groups only work on real data for this rowBlock.
         if(st < (stop_row - row))
         {
            // only works when numThreadsForRed is a power of 2
            for(int i = 0; i < workForEachThread; i++)
            {
               temp += partialSums[local_first_val + i*numThreadsForRed + threadInBlock];
            }

            // The last few values (the remainder of this row) also need to be aded in.
            int local_cur_val = local_first_val + numThreadsForRed*workForEachThread;
            if(threadInBlock < local_last_val - local_cur_val)
            {
               temp += partialSums[local_cur_val + threadInBlock];
            }
         }
         barrier(CLK_LOCAL_MEM_FENCE);
         partialSums[lid] = temp;

         // Step one of this two-stage reduction is done. Now each row has {numThreadsForRed}
         // values sitting in the local memory. This next step takes the first thread from
         // each of the adjacent-groups and uses it to walk through those values and reduce
         // them into a final output value for the row.
         temp = 0.;
         if(lid < (stop_row - row))
         {
#pragma unroll 4
            for(int i = 0; i < numThreadsForRed; i++)
            {
               temp += partialSums[lid*numThreadsForRed + i];
            }
            // All of our write-outs check to see if the output vector should first be zeroed.
            // If so, just do a write rather than a read-write. Measured to be a slight (~5%)
            // performance improvement.
            if (beta != 0.)
                out[local_row] = (beta * out[local_row]) + (alpha * temp);
            else
                out[local_row] = (alpha * temp);
         }
      }
      else
      {
          // In this case, we want to have each thread perform the reduction for a single row.
          // Essentially, this looks like performing CSR-Scalar, except it is computed out of local memory.
          // However, this reduction is also much faster than CSR-Scalar, because local memory
          // is designed for scatter-gather operations.
          // We need a while loop because there may be more rows than threads in the WG.
         while(local_row < stop_row)
         {
            int local_first_val = (rowPtrs[local_row] - rowPtrs[row]);
            int local_last_val = rowPtrs[local_row + 1] - rowPtrs[row];
            FPTYPE temp = 0;
            for (int local_cur_val = local_first_val; local_cur_val < local_last_val; local_cur_val++)
            {
               temp += partialSums[local_cur_val];
            }

            // After you've done the reduction into the temp register,
            // put that into the output for each row.
            if (beta != 0.)
                out[local_row] = (beta * out[local_row]) + (alpha * temp);
            else
                out[local_row] = (alpha * temp);
            local_row += get_local_size(0);
         }
      }
   }
   else
   {
       // In CSR-Vector, we may have more than one workgroup calculating this row
	   // or one workgroup calculating two rows.
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
	   size_t first_wg_in_row = gid - (rowBlocks[gid] & ((1UL << WGBITS) - 1UL));
       unsigned int compare_value = rowBlocks[gid] & (1UL << WGBITS);

       // Bit 24 in the first workgroup is the flag that everyone waits on.
	   if(gid == first_wg_in_row && lid == 0)
	   {
            // The first workgroup handles the output initialization.
            if (beta != 0.)
                out[row] *= beta;
            else
                out[row] = 0.;
            // We currently have, at most, two rows in a CSR-Vector calculation.
            // If we have two, we need to initialize the second output as well.
            if(stop_row - row == 2)
            {
                if (beta != 0.)
                    out[row+1] *= beta;
                else
                    out[row+1] = 0.;
            }
            atom_xor(&rowBlocks[first_wg_in_row], (1UL << WGBITS)); // Release other workgroups.
       }
       // For every other workgroup, bit 24 holds the value they wait on.
       // If your bit 24 == first_wg's bit 24, you spin loop.
       // The first workgroup will eventually flip this bit, and you can move forward.
       barrier(CLK_GLOBAL_MEM_FENCE);
	   while(gid != first_wg_in_row &&
             lid==0 &&
             ((atom_max(&rowBlocks[first_wg_in_row], 0UL) & (1UL << WGBITS)) == compare_value));
       barrier(CLK_GLOBAL_MEM_FENCE);

       // After you've passed the barrier, update your local flag to make sure that
       // the next time through, you know what to wait on.
       if (gid != first_wg_in_row && lid == 0)
           rowBlocks[gid] ^= (1UL << WGBITS);

       unsigned int myRow = row; // Adding another variable for row as it is getting updated below
       char iteration = 0;

       // CSR-Vector will always do at least one iteration.
       // All but the final workgroup in a long-row collaboration have the same start_row
       // and stop_row. They only run for one iteration.
       // If this workgroup is operating on multiple rows (because CSR-Stream is poor for small
       // #s of rows), then it is (currently) not part of a collaboration on a long
       // row. As such, it needs to iterate until it reaches the stop_row.
       // We don't check <= stop_row because of the potential for unsigned overflow.
		while (iteration == 0 || myRow < stop_row)
		{
            // Get the "workgroup within this long row" ID out of the bottom bits of the row block.
            // If this is the only workgroup working on this row, this will be zero, so still correct.
			unsigned int wg = rowBlocks[gid] & ((1 << WGBITS) - 1);

            // Any workgroup only calculates, at most, BLOCKSIZE items in this row.
            // If there are more items in this row, we assign more workgroups.
            unsigned int vecStart = rowPtrs[myRow] + (wg * BLOCKSIZE);
			unsigned int vecEnd = (rowPtrs[myRow + 1] > vecStart + BLOCKSIZE) ? vecStart + BLOCKSIZE : rowPtrs[myRow+1];

            // Load in a bunch of partial results into your register space, rather than LDS (no contention)
            // Then dump the partially reduced answers into the LDS for inter-work-item reduction.
            // Using a long induction variable to make sure unsigned int overflow doesn't break things.
			FPTYPE mySum = 0.;
			for (long j = vecStart + lid; j < vecEnd; j+=WGSIZE)
			 {
				 unsigned int col = cols[(unsigned int)j];
				 mySum += vals[(unsigned int)j] * vec[col];
			 }
			 partialSums[lid] = mySum;
			 barrier(CLK_LOCAL_MEM_FENCE);

			 // Reduce partial sums
			 // Needs to be modified if there is a change in the # of work-items per workgroup.
			 if(lid < 64) partialSums[lid] += partialSums[lid + 64] + partialSums[lid + 128] + partialSums[lid + 192];
			 barrier(CLK_LOCAL_MEM_FENCE);
			 if(lid < 16) partialSums[lid] += partialSums[lid + 16] + partialSums[lid + 32] + partialSums[lid + 48];
			 barrier(CLK_LOCAL_MEM_FENCE);
			 if(lid < 4) partialSums[lid] += partialSums[lid + 4] + partialSums[lid + 8] + partialSums[lid + 12];
			 barrier(CLK_LOCAL_MEM_FENCE);
			 if(lid < 1) partialSums[lid] += partialSums[lid + 1] + partialSums[lid + 2] + partialSums[lid + 3];
			 barrier(CLK_LOCAL_MEM_FENCE);

			 if (lid == 0)
				 atomic_add_float(&out[myRow], (alpha * partialSums[0]));

			 // CSR-VECTOR on workgroups for two rows which are inefficient for CSR-Stream
			 myRow++;
             iteration++;
		}
   }
}
)"
