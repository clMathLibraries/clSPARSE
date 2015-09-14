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
// typedef double   vT;
#define vT float

__kernel
void copyCt2C_Single(__global const int   *d_csrRowPtrC,
                     __global       int   *d_csrColIndC,
                     __global       vT *d_csrValC,
                     __global const int   *d_csrRowPtrCt,
                     __global const int   *d_csrColIndCt,
                     __global const vT *d_csrValCt,
                     __global const int   *d_queue,
                     const          int    size,
                     const          int    d_queue_offset)
{
    int global_id = get_global_id(0);

    bool valid = (global_id < size);

    int row_id = valid ? d_queue[TUPLE_QUEUE * (d_queue_offset + global_id)] : 0;

    int Ct_base_start = valid ? d_queue[TUPLE_QUEUE * (d_queue_offset + global_id) + 1] : 0; //d_csrRowPtrCt[row_id] : 0;
    int C_base_start  = valid ? d_csrRowPtrC[row_id] : 0;

    int colC   = valid ? d_csrColIndCt[Ct_base_start] : 0;
    vT valC = valid ? d_csrValCt[Ct_base_start] : 0.0f;

    if (valid)
    {
        d_csrColIndC[C_base_start] = colC;
        d_csrValC[C_base_start]    = valC;
    }
}

__kernel
void copyCt2C_Loopless(__global const int   *d_csrRowPtrC,
                       __global       int   *d_csrColIndC,
                       __global       vT *d_csrValC,
                       __global const int   *d_csrRowPtrCt,
                       __global const int   *d_csrColIndCt,
                       __global const vT *d_csrValCt,
                       __global const int   *d_queue,
                       const          int    d_queue_offset)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);

    int row_id = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id)];

    int Ct_base_start = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id) + 1] + local_id; //d_csrRowPtrCt[row_id] + local_id;
    int C_base_start  = d_csrRowPtrC[row_id]  + local_id;
    int C_base_stop   = d_csrRowPtrC[row_id + 1];

    if (C_base_start < C_base_stop)
    {
        d_csrColIndC[C_base_start] = d_csrColIndCt[Ct_base_start];
        d_csrValC[C_base_start]    = d_csrValCt[Ct_base_start];
    }
}

__kernel
void copyCt2C_Loop(__global const int   *d_csrRowPtrC,
                   __global       int   *d_csrColIndC,
                   __global       vT *d_csrValC,
                   __global const int   *d_csrRowPtrCt,
                   __global const int   *d_csrColIndCt,
                   __global const vT *d_csrValCt,
                   __global const int   *d_queue,
                   const          int    d_queue_offset)
{
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int local_size = get_local_size(0);

    int row_id = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id)];

    int Ct_base_start = d_queue[TUPLE_QUEUE * (d_queue_offset + group_id) + 1]; //d_csrRowPtrCt[row_id];
    int C_base_start  = d_csrRowPtrC[row_id];
    int C_base_stop   = d_csrRowPtrC[row_id + 1];
    int stride        = C_base_stop - C_base_start;

    bool valid;

    int loop = ceil((float)stride / (float)local_size);

    C_base_start  += local_id;
    Ct_base_start += local_id;

    for (int i = 0; i < loop; i++)
    {
        valid = (C_base_start < C_base_stop);

        if (valid)
        {
            d_csrColIndC[C_base_start] = d_csrColIndCt[Ct_base_start];
            d_csrValC[C_base_start]    = d_csrValCt[Ct_base_start];
        }

        C_base_start += local_size;
        Ct_base_start += local_size;
    }
}


)"
