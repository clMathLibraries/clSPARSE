R"(
/* ************************************************************************
* The MIT License (MIT)
* Copyright 2014-2015 University of Copenhagen
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

//#define GROUPSIZE_256 256

__kernel
void compute_nnzCt_kernel(
        __global const int *d_csrRowPtrA,
        __global const int *d_csrColIndA,
        __global const int *d_csrRowPtrB,
        __global int *d_csrRowPtrCt,
        const int m)
{
    int global_id = get_global_id(0);
    int start, stop, index, strideB, row_size_Ct = 0;

    if (global_id < m)
    {
        start = d_csrRowPtrA[global_id];
        stop = d_csrRowPtrA[global_id + 1];

        for (int i = start; i < stop; i++)
        {
            index = d_csrColIndA[i];
            strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
            row_size_Ct += strideB;
        }

        d_csrRowPtrCt[global_id] = row_size_Ct;
    }

    if (global_id == 0)
        d_csrRowPtrCt[m] = 0;
}

)"
