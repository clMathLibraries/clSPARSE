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
 

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define TUPLE_QUEUE 6
// typedef double vT;
#define vT float

__kernel
void ESC_0(__global const int   *d_queue,
           __global int         *d_csrRowPtrC,
           const int             queue_size,
           const int             queue_offset)
{
    int global_id = get_global_id(0); //blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < queue_size)
    {
        int row_id = d_queue[TUPLE_QUEUE * (queue_offset + global_id)];
        d_csrRowPtrC[row_id] = 0;
    }
}

__kernel
void ESC_1(__global const int   *d_queue,
           __global const int   *d_csrRowPtrA,
           __global const int   *d_csrColIndA,
           __global const vT    *d_csrValA,
           __global const int   *d_csrRowPtrB,
           __global const int   *d_csrColIndB,
           __global const vT    *d_csrValB,
           __global int         *d_csrRowPtrC,
           __global const int   *d_csrRowPtrCt,
           __global int         *d_csrColIndCt,
           __global vT          *d_csrValCt,
           const int             queue_size,
           const int             queue_offset)
{
    int global_id = get_global_id(0); //blockIdx.x * blockDim.x + threadIdx.x;

    if (global_id < queue_size)
    {
        int row_id = d_queue[TUPLE_QUEUE * (queue_offset + global_id)];
        d_csrRowPtrC[row_id] = 1;

        int base_index = d_queue[TUPLE_QUEUE * (queue_offset + global_id) + 1]; //d_csrRowPtrCt[row_id];

        int col_index_A_start = d_csrRowPtrA[row_id];
        int col_index_A_stop = d_csrRowPtrA[row_id+1];

        for (int col_index_A = col_index_A_start; col_index_A < col_index_A_stop; col_index_A++)
        {
            int row_id_B = d_csrColIndA[col_index_A];
            int col_index_B = d_csrRowPtrB[row_id_B];

            if (col_index_B == d_csrRowPtrB[row_id_B+1])
                continue;

            vT value_A  = d_csrValA[col_index_A];

            d_csrColIndCt[base_index] = d_csrColIndB[col_index_B];
            d_csrValCt[base_index] = d_csrValB[col_index_B] * value_A;

            break;
        }
    }
}


)"
