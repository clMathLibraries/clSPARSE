R"(
/* ************************************************************************
* The MIT License (MIT)
* Copyright 2014-2015 weifengliu
*  Permission is hereby granted, free of charge, to any person obtaining a copy
*  of this software and associated documentation files (the "Software"), to deal
*  in the Software without restriction, including without limitation the rights
*  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
*  copies of the Software, and to permit persons to whom the Software is
*  furnished to do so, subject to the following conditions:
 
*  The above copyright notice and this permission notice shall be included in
*  all copies or substantial portions of the Software.
 
*  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
*  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
*  THE SOFTWARE.
* ************************************************************************ */
/* ************************************************************************
*  < A CUDA/OpenCL General Sparse Matrix-Matrix Multiplication Program >
*  
*  < See papers:
*  1. Weifeng Liu and Brian Vinter, "A Framework for General Sparse 
*      Matrix-Matrix Multiplication on GPUs and Heterogeneous 
*      Processors," Journal of Parallel and Distributed Computing, 2015.
*  2. Weifeng Liu and Brian Vinter, "An Efficient GPU General Sparse
*      Matrix-Matrix Multiplication for Irregular Data," Parallel and
*      Distributed Processing Symposium, 2014 IEEE 28th International
*      (IPDPS '14), pp.370-381, 19-23 May 2014.
*  for details. >
* ************************************************************************ */ 

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

#define TUPLE_QUEUE 6
// typedef double   vT;
#define vT float

inline
void coex(__local  int      *keyA,
          __local  vT    *valA,
          __local  int      *keyB,
          __local  vT    *valB,
          const int          dir)
{
    int t;
    vT v;

    if ((*keyA > *keyB) == dir)
    {
        t = *keyA;
        *keyA = *keyB;
        *keyB = t;
        v = *valA;
        *valA = *valB;
        *valB = v;
    }
}


inline
void bitonic(__local int    *s_key,
             __local vT     *s_val,
                    int             stage,
                    int             passOfStage,
                    int             local_id,
                    int             local_counter)
{
    int sortIncreasing = 1;
    int pairDistance = 1 << (stage - passOfStage);
    int blockWidth   = 2 * pairDistance;
    int leftId = (local_id % pairDistance) + (local_id / pairDistance) * blockWidth;
    int rightId = leftId + pairDistance;
    int sameDirectionBlockWidth = 1 << stage;
    if((local_id/sameDirectionBlockWidth) % 2 == 1)
        sortIncreasing = 1 - sortIncreasing;
    int  leftElement  = s_key[leftId];          // index_type
    int  rightElement = s_key[rightId];         // index_type
    vT   leftElement_val  = s_val[leftId];      // value_type
    vT   rightElement_val = s_val[rightId];     // value_type
    int  greater;         // index_type
    int  lesser;          // index_type
    vT   greater_val;     // value_type
    vT   lesser_val;      // value_type
    if(leftElement > rightElement)
    {
        greater = leftElement;
        lesser  = rightElement;
        greater_val = leftElement_val;
        lesser_val  = rightElement_val;
    }
    else
    {
        greater = rightElement;
        lesser  = leftElement;
        greater_val = rightElement_val;
        lesser_val  = leftElement_val;
    }
    if(sortIncreasing)
    {
        s_key[leftId]  = lesser;
        s_key[rightId] = greater;
        s_val[leftId]  = lesser_val;
        s_val[rightId] = greater_val;
    }
    else
    {
        s_key[leftId]  = greater;
        s_key[rightId] = lesser;
        s_val[leftId]  = greater_val;
        s_val[rightId] = lesser_val;
    }
}
inline
void bitonicsort(__local int   *s_key,
                 __local vT    *s_val,
                 int            arrayLength)
{
    int local_id = get_local_id(0);
    int numStages = 0;
    for(int temp = arrayLength; temp > 1; temp >>= 1)
    {
        ++numStages;  // arrayLength = 2^numStages;
    }
    for (int stage = 0; stage < numStages; ++stage)
    {
        for (int passOfStage = 0; passOfStage <= stage; ++passOfStage)
        {
            bitonic(s_key, s_val, stage, passOfStage, local_id, arrayLength);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
}


inline
void scan_32(__local volatile short *s_scan)
{
    int local_id = get_local_id(0);
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    if (local_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[31] += s_scan[15]; s_scan[32] = s_scan[31]; s_scan[31] = 0; temp = s_scan[15]; s_scan[15] = 0; s_scan[31] += temp; }
    if (local_id < 2)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 2 * baseai - 1;  bi = 2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}

inline
void scan_64(__local volatile short *s_scan)
{
    int local_id = get_local_id(0);
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[63] += s_scan[31]; s_scan[64] = s_scan[63]; s_scan[63] = 0; temp = s_scan[31]; s_scan[31] = 0; s_scan[63] += temp; }
    if (local_id < 2)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 4 * baseai - 1;  bi = 4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}

inline
void scan_128(__local volatile short *s_scan)
{
    int local_id = get_local_id(0);
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[127] += s_scan[63]; s_scan[128] = s_scan[127]; s_scan[127] = 0; temp = s_scan[63]; s_scan[63] = 0; s_scan[127] += temp; }
    if (local_id < 2)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 8 * baseai - 1;  bi = 8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 32) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}

inline
void scan_256(__local volatile short *s_scan)
{
    int local_id = get_local_id(0);
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[255] += s_scan[127]; s_scan[256] = s_scan[255]; s_scan[255] = 0; temp = s_scan[127]; s_scan[127] = 0; s_scan[255] += temp; }
    if (local_id < 2)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 16 * baseai - 1;  bi = 16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 32) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}

inline
void scan_512(__local volatile short *s_scan)
{
    int local_id = get_local_id(0);
    int ai, bi;
    int baseai = 1 + 2 * local_id;
    int basebi = baseai + 1;
    short temp;

    ai = baseai - 1;     bi = basebi - 1;     s_scan[bi] += s_scan[ai];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64)  { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   s_scan[bi] += s_scan[ai]; }
    if (local_id == 0) { s_scan[511] += s_scan[255]; s_scan[512] = s_scan[511]; s_scan[511] = 0; temp = s_scan[255]; s_scan[255] = 0; s_scan[511] += temp; }
    if (local_id < 2)  { ai = 128 * baseai - 1;  bi = 128 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 4)  { ai = 64 * baseai - 1;  bi = 64 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 8)  { ai = 32 * baseai - 1;  bi = 32 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 16) { ai =  16 * baseai - 1;  bi =  16 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    if (local_id < 32) { ai =  8 * baseai - 1;  bi =  8 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 64) { ai =  4 * baseai - 1;  bi =  4 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < 128) { ai =  2 * baseai - 1;  bi =  2 * basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;}
    barrier(CLK_LOCAL_MEM_FENCE);
    ai = baseai - 1;   bi = basebi - 1;   temp = s_scan[ai]; s_scan[ai] = s_scan[bi]; s_scan[bi] += temp;
}
)"

R"(
inline
void compression_scan(__local volatile short *s_scan,
                      __local int            *s_key,
                      __local vT          *s_val,
                      const int               local_counter,
                      const int               local_size,
                      const int               local_id,
                      const int               local_id_halfwidth)
{
    // compression - prefix sum
    bool duplicate = 1;
    bool duplicate_halfwidth = 1;

    // generate bool value in registers
    if (local_id < local_counter && local_id > 0)
        duplicate = (s_key[local_id] != s_key[local_id - 1]);
    if (local_id_halfwidth < local_counter)
        duplicate_halfwidth = (s_key[local_id_halfwidth] != s_key[local_id_halfwidth - 1]);

    // copy bool values from register to local memory (s_scan)
    s_scan[local_id]                    = duplicate;
    s_scan[local_id_halfwidth]          = duplicate_halfwidth;
    barrier(CLK_LOCAL_MEM_FENCE);

    // in-place exclusive prefix-sum scan on s_scan
    switch (local_size)
    {
    case 16:
        scan_32(s_scan);
        break;
    case 32:
        scan_64(s_scan);
        break;
    case 64:
        scan_128(s_scan);
        break;
    case 128:
        scan_256(s_scan);
        break;
    case 256:
        scan_512(s_scan);
        break;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute final position and final value in registers
    int   move_pointer;
    short final_position, final_position_halfwidth;
    int   final_key,      final_key_halfwidth;
    vT final_value,    final_value_halfwidth;

    if (local_id < local_counter && duplicate == 1)
    {
        final_position = s_scan[local_id];
        final_key = s_key[local_id];
        final_value = s_val[local_id];
        move_pointer = local_id + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1])
        {
            final_value += s_val[move_pointer];
            move_pointer++;
        }
    }

    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        final_position_halfwidth = s_scan[local_id_halfwidth];
        final_key_halfwidth = s_key[local_id_halfwidth];
        final_value_halfwidth = s_val[local_id_halfwidth];
        move_pointer = local_id_halfwidth + 1;

        while (s_scan[move_pointer] == s_scan[move_pointer + 1] && move_pointer < 2 * local_size)
        {
            final_value_halfwidth += s_val[move_pointer];
            move_pointer++;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // write final_positions and final_values to s_key and s_val
    if (local_id < local_counter && duplicate == 1)
    {
        s_key[final_position] = final_key;
        s_val[final_position] = final_value;
    }
    if (local_id_halfwidth < local_counter && duplicate_halfwidth == 1)
    {
        s_key[final_position_halfwidth] = final_key_halfwidth;
        s_val[final_position_halfwidth] = final_value_halfwidth;
    }
}
)"

R"(
__kernel
void ESC_bitonic_scan(__global const int      *d_queue,
                      __global const int      *d_csrRowPtrA,
                      __global const int      *d_csrColIndA,
                      __global const vT    *d_csrValA,
                      __global const int      *d_csrRowPtrB,
                      __global const int      *d_csrColIndB,
                      __global const vT    *d_csrValB,
                      __global int            *d_csrRowPtrC,
                      __global const int      *d_csrRowPtrCt,
                      __global int            *d_csrColIndCt,
                      __global vT          *d_csrValCt,
                      __local  int            *s_key,
                      __local  vT          *s_val,
                      __local  volatile short *s_scan,
                      const int                queue_offset,
                      const int                n)
{
    int local_id = get_local_id(0); //local_id;
    int group_id = get_group_id(0); //blockIdx.x;
    int local_size = get_local_size(0);
    int width = local_size * 2;

    int i, local_counter = 0;
    int strideB, local_offset, global_offset;
    int invalid_width;
    int local_id_halfwidth = local_id + local_size;

    int row_id_B; // index_type

    int row_id;// index_type
    row_id = d_queue[TUPLE_QUEUE * (queue_offset + group_id)];

    int start_col_index_A, stop_col_index_A;  // index_type
    int start_col_index_B, stop_col_index_B;  // index_type
    vT value_A;                            // value_type

    start_col_index_A = d_csrRowPtrA[row_id];
    stop_col_index_A  = d_csrRowPtrA[row_id + 1];

    // i is both col index of A and row index of B
    for (i = start_col_index_A; i < stop_col_index_A; i++)
    {
        row_id_B = d_csrColIndA[i];
        value_A  = d_csrValA[i];

        start_col_index_B = d_csrRowPtrB[row_id_B];
        stop_col_index_B  = d_csrRowPtrB[row_id_B + 1];

        strideB = stop_col_index_B - start_col_index_B;

        if (local_id < strideB)
        {
            local_offset = local_counter + local_id;
            global_offset = start_col_index_B + local_id;

            s_key[local_offset] = d_csrColIndB[global_offset];
            s_val[local_offset] = d_csrValB[global_offset] * value_A;
        }

        if (local_id_halfwidth < strideB)
        {
            local_offset = local_counter + local_id_halfwidth;
            global_offset = start_col_index_B + local_id_halfwidth;

            s_key[local_offset] = d_csrColIndB[global_offset];
            s_val[local_offset] = d_csrValB[global_offset] * value_A;
        }

        local_counter += strideB;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    invalid_width = width - local_counter;

    // to meet 2^N, set the rest elements to n (number of columns of C)
    if (local_id < invalid_width)
        s_key[local_counter + local_id] = n;
    //if (local_id_halfwidth < invalid_width)
    //    s_key[local_counter + local_id_halfwidth] = n;
    barrier(CLK_LOCAL_MEM_FENCE);

    // bitonic sort
    bitonicsort(s_key, s_val, width);
    barrier(CLK_LOCAL_MEM_FENCE);

    // compression - scan
    compression_scan(s_scan, s_key, s_val, local_counter,
                     local_size, local_id, local_id_halfwidth);
    barrier(CLK_LOCAL_MEM_FENCE);

    local_counter = s_scan[width] - invalid_width;
    if (local_id == 0)
        d_csrRowPtrC[row_id] = local_counter;

    // write compressed lists to global mem
    int row_offset = d_queue[TUPLE_QUEUE * (queue_offset + group_id) + 1]; //d_csrRowPtrCt[row_id];

    if (local_id < local_counter)
    {
        global_offset = row_offset + local_id;

        d_csrColIndCt[global_offset] = s_key[local_id];
        d_csrValCt[global_offset] = s_val[local_id];
    }
    if (local_id_halfwidth < local_counter)
    {
        global_offset = row_offset + local_id_halfwidth;

        d_csrColIndCt[global_offset] = s_key[local_id_halfwidth];
        d_csrValCt[global_offset] = s_val[local_id_halfwidth];
    }
}
)"
