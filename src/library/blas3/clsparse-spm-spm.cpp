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


#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/clsparse-internal.hpp"

#include <cmath>

#define GROUPSIZE_256 256
#define TUPLE_QUEUE 6
#define NUM_SEGMENTS 128
//#define WARPSIZE_NV_2HEAP 64
#define value_type float
#define index_type int 
#define MERGEPATH_LOCAL     0
#define MERGEPATH_LOCAL_L2  1
#define MERGEPATH_GLOBAL    2
#define MERGELIST_INITSIZE 256
#define BHSPARSE_SUCCESS 0
 
#define CHECKMAL(a, b) \
    if(a==NULL) \
    {     \
        std::cerr << "Malloc Error:" << b << std::endl; \
        exit(1); \
    }
 
using namespace std;

int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m);

clsparseStatus compute_nnzCt(int _m, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrRowPtrCt, clsparseControl control){
    
     const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_computeNnzCt_kernels", "compute_nnzCt_kernel", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    int num_threads = GROUPSIZE_256;
    int num_blocks = ceil((double)_m / (double)num_threads);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);

    kWrapper << csrRowPtrA << csrColIndA << csrRowPtrB << csrRowPtrCt << _m;
    
    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
    
 }
 

 
clsparseStatus compute_nnzC_Ct_0(int num_threads, int num_blocks, int j, int counter, int position, cl_mem queue_one, cl_mem csrRowPtrC, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_ESC_0_1_kernels", "ESC_0", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];
    
    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrC << counter << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
    
}

clsparseStatus compute_nnzC_Ct_1(int num_threads, int num_blocks, int j, int counter, int position, cl_mem queue_one, 
                      cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB, cl_mem csrColIndB, 
                      cl_mem csrValB, cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem csrValCt, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_ESC_0_1_kernels", "ESC_1", params);

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];


    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB <<  csrRowPtrC << csrRowPtrCt << csrColIndCt << csrValCt << counter << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
    
}

clsparseStatus compute_nnzC_Ct_2heap_noncoalesced_local(int num_threads, int num_blocks, int j, int counter, int position, 
                                             cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, 
                                             cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, 
                                             cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem csrValCt, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_ESC_2heap_kernels", "ESC_2heap_noncoalesced_local", params);

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC 
             << csrRowPtrCt << csrColIndCt << csrValCt << cl::__local(j*num_threads * sizeof(int) ) << cl::__local(j*num_threads * sizeof(float)) << counter << position;
    
    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

clsparseStatus compute_nnzC_Ct_bitonic_scan(int num_threads, int num_blocks, int j, int position, cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB, 
                                 cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem csrValCt, int _n, clsparseControl control)
{
    

    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_ESC_bitonic_kernels", "ESC_bitonic_scan", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];
    
    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    int buffer_size = 2 * num_threads;

    KernelWrap kWrapper(kernel);
    kWrapper << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC << csrRowPtrCt
                          << csrColIndCt << csrValCt << cl::__local(buffer_size * sizeof(int)) << cl::__local(buffer_size * sizeof(float)) << cl::__local((buffer_size+1) * sizeof(short)) << position << _n;
    
    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;

}

clsparseStatus compute_nnzC_Ct_mergepath(int num_threads, int num_blocks, int j, int mergebuffer_size, int position, int *count_next, int mergepath_location, 
                                         cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, 
                                         cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem *csrColIndCt, cl_mem *csrValCt, int *_nnzCt, int m, int *_h_queue_one, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;

    cl::Kernel kernel1  = KernelCache::get(control->queue,"SpGEMM_EM_kernels", "EM_mergepath", params);
    cl::Kernel kernel2  = KernelCache::get(control->queue,"SpGEMM_EM_kernels", "EM_mergepath_global", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];
    
    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];
    
    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status;

    if (mergepath_location == MERGEPATH_LOCAL)
    {
       KernelWrap kWrapper1(kernel1);
       kWrapper1 << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB <<  csrValB << csrRowPtrC 
                << csrRowPtrCt <<  *csrColIndCt <<  *csrValCt << cl::__local((mergebuffer_size) * sizeof(int)) << cl::__local((mergebuffer_size) * sizeof(float)) <<  cl::__local((num_threads+1) * sizeof(short)) <<  position << mergebuffer_size << cl::__local(sizeof(cl_int)   * (num_threads + 1)) << cl::__local(sizeof(cl_int)   * (num_threads + 1));

                           
    status = kWrapper1.run(control, global, local);
    
       if (status != CL_SUCCESS)
       {
          return clsparseInvalidKernelExecution;
       }
    
    }
    else if (mergepath_location == MERGEPATH_GLOBAL)
    {
       int mergebuffer_size_local = 2304;
    
       KernelWrap kWrapper2(kernel2);
       kWrapper2 << queue_one << csrRowPtrA << csrColIndA << csrValA << csrRowPtrB << csrColIndB << csrValB << csrRowPtrC
                          << csrRowPtrCt << *csrColIndCt <<  *csrValCt << cl::__local((mergebuffer_size_local) * sizeof(int)) << cl::__local((mergebuffer_size_local) * sizeof(float)) << cl::__local(( num_threads+1) * sizeof(short)) << position << mergebuffer_size_local << cl::__local(sizeof(cl_int)   * (num_threads + 1)) << cl::__local(sizeof(cl_int)   * (num_threads + 1));

    
       status = kWrapper2.run(control, global, local);
    
       if (status != CL_SUCCESS)
       {
         return clsparseInvalidKernelExecution;
       }
    
    }

    int temp_queue [6] = {0, 0, 0, 0, 0, 0};
    int counter = 0;
    int temp_num = 0;

    status = clEnqueueReadBuffer(control->queue(),
                                 queue_one,
                                  1,
                                     0,
                                     TUPLE_QUEUE * m * sizeof(int),
                                     _h_queue_one,
                                     0,
                                     0,
                                     0);

    for (int i = position; i < position + num_blocks; i++)
    {
        if (_h_queue_one[TUPLE_QUEUE * i + 2] != -1)
        {
            temp_queue[0] = _h_queue_one[TUPLE_QUEUE * i]; // row id
            if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
            {
                //temp_queue[1] = _nnzCt + counter * mergebuffer_size * 2; // new start address
                int accum = 0;
                switch (mergebuffer_size)
                {
                case 256:
                    accum = 512;
                    break;
                case 512:
                    accum = 1024;
                    break;
                case 1024:
                    accum = 2048;
                    break;
                case 2048:
                    accum = 2304;
                    break;
                case 2304:
                    accum = 2 * (2304 * 2);
                    break;
                }

                temp_queue[1] = *_nnzCt + counter * accum; // new start address
            }
            else if (mergepath_location == MERGEPATH_GLOBAL)
                temp_queue[1] = *_nnzCt + counter * (2 * (mergebuffer_size + 2304)); 
            temp_queue[2] = _h_queue_one[TUPLE_QUEUE * i + 2]; // merged size
            temp_queue[3] = _h_queue_one[TUPLE_QUEUE * i + 3]; // i
            temp_queue[4] = _h_queue_one[TUPLE_QUEUE * i + 4]; // k
            temp_queue[5] = _h_queue_one[TUPLE_QUEUE * i + 1]; // old start address

            _h_queue_one[TUPLE_QUEUE * i]     = _h_queue_one[TUPLE_QUEUE * (position + counter)];     // row id
            _h_queue_one[TUPLE_QUEUE * i + 1] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 1]; // new start address
            _h_queue_one[TUPLE_QUEUE * i + 2] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 2]; // merged size
            _h_queue_one[TUPLE_QUEUE * i + 3] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 3]; // i
            _h_queue_one[TUPLE_QUEUE * i + 4] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 4]; // k
            _h_queue_one[TUPLE_QUEUE * i + 5] = _h_queue_one[TUPLE_QUEUE * (position + counter) + 5]; // old start address

            _h_queue_one[TUPLE_QUEUE * (position + counter)]     = temp_queue[0]; // row id
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 1] = temp_queue[1]; // new start address
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 2] = temp_queue[2]; // merged size
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 3] = temp_queue[3]; // i
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 4] = temp_queue[4]; // k
            _h_queue_one[TUPLE_QUEUE * (position + counter) + 5] = temp_queue[5]; // old start address

            counter++;
            temp_num += _h_queue_one[TUPLE_QUEUE * i + 2];
        }
    }

    status = clEnqueueWriteBuffer(control->queue(),
                                      queue_one,
                                      1,
                                      0,
                                      TUPLE_QUEUE * m * sizeof(int),
                                      _h_queue_one,
                                      0,
                                      0,
                                      0);

    //*
    if (counter > 0)
    {
        int nnzCt_new;
        if (mergepath_location == MERGEPATH_LOCAL || mergepath_location == MERGEPATH_LOCAL_L2)
        {
            //nnzCt_new = _nnzCt + counter * mergebuffer_size * 2; // new start address
            int accum = 0;
            switch (mergebuffer_size)
            {
            case 256:
                accum = 512;
                break;
            case 512:
                accum = 1024;
                break;
            case 1024:
                accum = 2048;
                break;
            case 2048:
                accum = 2304;
                break;
            case 2304:
                accum = 2 * (2304 * 2);
                break;
            }

            nnzCt_new = *_nnzCt + counter * accum;
        }
        else if (mergepath_location == MERGEPATH_GLOBAL)
        nnzCt_new = *_nnzCt + counter * (2 * (mergebuffer_size + 2304));
        //cout << endl << "    ==> nnzCt_new = " << nnzCt_new << endl;

        cl::Context cxt = control->getContext();
    
    cl_mem  csrColIndCt_new = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzCt_new * sizeof( cl_int ), NULL, NULL );
        cl_mem  csrValCt_new    = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzCt_new * sizeof( cl_float ), NULL, NULL );

        clEnqueueCopyBuffer (	control->queue(),
                                *csrColIndCt,
                                csrColIndCt_new,
                                0,
                                0,
                                sizeof(cl_int)*(*_nnzCt),
                                0,
                                NULL,
                                NULL);
        
    clEnqueueCopyBuffer (	control->queue(),
                               *csrValCt,
                                csrValCt_new,
                                0,
                                0,
                                sizeof(cl_float)*(*_nnzCt),
                                0,
                                NULL,
                                NULL);
        
    clReleaseMemObject (*csrColIndCt);
        clReleaseMemObject (*csrValCt);

        *csrColIndCt = csrColIndCt_new;
        *csrValCt = csrValCt_new;
        
        *_nnzCt = nnzCt_new;
    }
    // */

    *count_next = counter;

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
    
}
 
clsparseStatus compute_nnzC_Ct_opencl(int *_h_counter_one, cl_mem queue_one, cl_mem csrRowPtrA, cl_mem csrColIndA, cl_mem csrValA, cl_mem csrRowPtrB, cl_mem csrColIndB, cl_mem csrValB, cl_mem csrRowPtrC, cl_mem csrRowPtrCt, cl_mem *csrColIndCt, cl_mem *csrValCt, int _n, int _nnzCt, int m, int *queue_one_h, clsparseControl control)
{
    //int err = 0;
    int counter = 0;
    
    clsparseStatus run_status;
    
    for (int j = 0; j < NUM_SEGMENTS; j++)
    {
        counter = _h_counter_one[j+1] - _h_counter_one[j];
        if (counter != 0)
        {

            if (j == 0)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks = ceil((double)counter / (double)num_threads);

                run_status = compute_nnzC_Ct_0(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrC, control);
            }
            else if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks = ceil((double)counter / (double)num_threads);

                run_status = compute_nnzC_Ct_1(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, control);
            }
            else if (j > 1 && j <= 32)
            {
              int num_threads = 64; //WARPSIZE_NV_2HEAP;
                int num_blocks = ceil((double)counter / (double)num_threads);
                run_status = compute_nnzC_Ct_2heap_noncoalesced_local(num_threads, num_blocks, j, counter, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, control);
            }
            else if (j > 32 && j <= 64)
            {
                int num_threads = 32;
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, _n, control);
            }
            else if (j > 64 && j <= 122)
            {
                int num_threads = 64;
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, _n, control);
            }
            else if (j == 123)
            {
                int num_threads = 128;
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, _n, control);
            }
            else if (j == 124)
            {
                int num_threads = 256;
                int num_blocks = counter;

                run_status = compute_nnzC_Ct_bitonic_scan(num_threads, num_blocks, j, _h_counter_one[j], queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, *csrColIndCt, *csrValCt, _n, control);
            }
            else if (j == 127)
            {
                int count_next = counter;
                int num_threads, num_blocks, mergebuffer_size;

                int num_threads_queue [5] = {64, 128, 256, 256, 256};
                int mergebuffer_size_queue [5] = {256, 512, 1024, 2048, 2304}; //{256, 464, 924, 1888, 3840};

                int queue_counter = 0;

                while (count_next > 0)
                {
                    num_blocks = count_next;

                    if (queue_counter < 5)
                    {
                        num_threads = num_threads_queue[queue_counter];
                        mergebuffer_size = mergebuffer_size_queue[queue_counter];

                        run_status = compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_LOCAL, queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, &_nnzCt, m, queue_one_h, control);

                        queue_counter++;
                    }
                    else
                    {
                        num_threads = num_threads_queue[4];
                        mergebuffer_size += mergebuffer_size_queue[4];
                        //cout << "    ==> doing merge on device mem, mergebuffer_size = " << mergebuffer_size << endl << endl;
                      
                        run_status = compute_nnzC_Ct_mergepath(num_threads, num_blocks, j, mergebuffer_size, _h_counter_one[j], &count_next, MERGEPATH_GLOBAL, queue_one, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt, csrColIndCt, csrValCt, &_nnzCt, m, queue_one_h, control);

                    }
                }

            }
      
        if (run_status != clsparseSuccess)
            {
               return clsparseInvalidKernelExecution;
            }
        }
    }
    
    return clsparseSuccess;

}


clsparseStatus copy_Ct_to_C_Single(int num_threads, int num_blocks, int local_size, int position, 
                        cl_mem csrValC, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValCt, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{

    int j = 1;

    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_copyCt2C_kernels", "copyCt2C_Single", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrValC << csrRowPtrCt << csrColIndCt << csrValCt << queue_one << local_size << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
}

clsparseStatus copy_Ct_to_C_Loopless(int num_threads, int num_blocks, int j, int position, 
                                     cl_mem csrValC, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValCt, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{
    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_copyCt2C_kernels", "copyCt2C_Loopless", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrValC << csrRowPtrCt << csrColIndCt << csrValCt << queue_one << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
    
}

clsparseStatus copy_Ct_to_C_Loop(int num_threads, int num_blocks, int j, int position, 
                                 cl_mem csrValC, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValCt, 
                                 cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{

    const std::string params = std::string() +
               "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type;
    
    
    cl::Kernel kernel = KernelCache::get(control->queue,"SpGEMM_copyCt2C_kernels", "copyCt2C_Loop", params);
    
    size_t szLocalWorkSize[1];
    size_t szGlobalWorkSize[1];

    szLocalWorkSize[0]  = num_threads;
    szGlobalWorkSize[0] = num_blocks * szLocalWorkSize[0];

    KernelWrap kWrapper(kernel);
    kWrapper << csrRowPtrC << csrColIndC << csrValC << csrRowPtrCt << csrColIndCt << csrValCt << queue_one << position;

    cl::NDRange local(szLocalWorkSize[0]);
    cl::NDRange global(szGlobalWorkSize[0]);

    cl_int status = kWrapper.run(control, global, local);
    
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
    
    return clsparseSuccess;
}


int copy_Ct_to_C_opencl(int *counter_one, cl_mem csrValC, cl_mem csrRowPtrC, cl_mem csrColIndC, cl_mem csrValCt, cl_mem csrRowPtrCt, cl_mem csrColIndCt, cl_mem queue_one, clsparseControl control)
{
    int counter = 0;
    
    clsparseStatus run_status;

    for (int j = 1; j < NUM_SEGMENTS; j++)
    {
        counter = counter_one[j+1] - counter_one[j];
        if (counter != 0)
        {
            if (j == 1)
            {
                int num_threads = GROUPSIZE_256;
                int num_blocks  = ceil((double)counter / (double)num_threads);
                run_status = copy_Ct_to_C_Single( num_threads, num_blocks, counter, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            }
            else if (j > 1 && j <= 32)
                run_status = copy_Ct_to_C_Loopless(   32, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 32 && j <= 64)
                run_status = copy_Ct_to_C_Loopless(   64, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 63 && j <= 96)
                run_status = copy_Ct_to_C_Loopless(   96, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j > 96 && j <= 122)
                run_status = copy_Ct_to_C_Loopless(  128, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 123)
                run_status = copy_Ct_to_C_Loopless(  256, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 124)
                run_status = copy_Ct_to_C_Loop( 256, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);
            else if (j == 127)
                run_status = copy_Ct_to_C_Loop( 256, counter, j, counter_one[j], csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt, csrColIndCt, queue_one, control);

            if (run_status != CL_SUCCESS)
            {
                return clsparseInvalidKernelExecution;
            }
        }
    }
    
    return clsparseSuccess;

}
 

 CLSPARSE_EXPORT clsparseStatus
        clsparseScsrSpGemm( 
        const clsparseCsrMatrix* sparseMatA,
        const clsparseCsrMatrix* sparseMatB,
              clsparseCsrMatrix* sparseMatC,
        const clsparseControl control )
{
    cl_int run_status;

    if (!clsparseInitialized)
    {
       return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
       return clsparseInvalidControlObject;
    }

    const clsparseCsrMatrixPrivate* matA = static_cast<const clsparseCsrMatrixPrivate*>(sparseMatA);
    const clsparseCsrMatrixPrivate* matB = static_cast<const clsparseCsrMatrixPrivate*>(sparseMatB);
    clsparseCsrMatrixPrivate* matC = static_cast<clsparseCsrMatrixPrivate*>(sparseMatC);

    int m  = matA->num_rows;
    int k1 = matA->num_cols;
    int k2 = matB->num_rows;
    int n  = matB->num_cols;
    int nnzA = matA->num_nonzeros;
    int nnzB = matB->num_nonzeros;
    
    if(k1 != k2)
    {
        std::cerr << "A.n and B.m don't match!" << std::endl; 
        return clsparseInvalidKernelExecution;
    }  
    
    cl_mem csrRowPtrA = matA->rowOffsets;
    cl_mem csrColIndA = matA->colIndices;
    cl_mem csrValA    = matA->values;
    cl_mem csrRowPtrB = matB->rowOffsets;
    cl_mem csrColIndB = matB->colIndices;
    cl_mem csrValB    = matB->values;
    
    cl::Context cxt = control->getContext();
    
    matC->rowOffsets = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof( cl_int ), NULL, &run_status );
    
    int pattern = 0;
    clEnqueueFillBuffer(control->queue(), matC->rowOffsets, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);
                        
    cl_mem csrRowPtrC = matC->rowOffsets;

    std::vector<int> csrRowPtrC_h(m + 1, 0);

    cl_mem csrRowPtrCt_d = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, (m + 1) * sizeof( cl_int ), NULL, &run_status );
    clEnqueueFillBuffer(control->queue(), csrRowPtrCt_d, &pattern, sizeof(cl_int), 0, (m + 1)*sizeof(cl_int), 0, NULL, NULL);

    std::vector<int> csrRowPtrCt_h(m + 1, 0);
    
    // STAGE 1
    compute_nnzCt(m, csrRowPtrA, csrColIndA, csrRowPtrB, csrColIndB, csrRowPtrCt_d, control);
    
    // statistics
    std::vector<int> counter(NUM_SEGMENTS, 0);

    std::vector<int> counter_one(NUM_SEGMENTS + 1, 0);

    std::vector<int> counter_sum(NUM_SEGMENTS + 1, 0);

    std::vector<int> queue_one(m * TUPLE_QUEUE, 0);
    
    cl_mem queue_one_d = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, TUPLE_QUEUE * m * sizeof(int), NULL, &run_status );
        
    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrCt_d,
                                     1,
                                     0,
                                     (m + 1)*sizeof(cl_int),
                                     csrRowPtrCt_h.data(),
                                     0,
                                     0,
                                     0);

    // STAGE 2 - STEP 1 : statistics
    int nnzCt = statistics(csrRowPtrCt_h.data(), counter.data(), counter_one.data(), counter_sum.data(), queue_one.data(), m);
    // STAGE 2 - STEP 2 : create Ct
    //cout << "nnzCt == " <<  nnzCt << endl; 
    
    cl_mem csrColIndCt = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzCt * sizeof( cl_int ), NULL, &run_status );
    cl_mem csrValCt    = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzCt * sizeof( cl_float ), NULL, &run_status );
    
    //copy queue_one
    run_status = clEnqueueWriteBuffer(control->queue(),
                                     queue_one_d,
                                     1,
                                     0,
                                     TUPLE_QUEUE * m * sizeof(int),
                                     queue_one.data(),
                                     0,
                                     0,
                                     0);
    
    // STAGE 3 - STEP 1 : compute nnzC and Ct
    compute_nnzC_Ct_opencl(counter_one.data(), queue_one_d, csrRowPtrA, csrColIndA, csrValA, csrRowPtrB, csrColIndB, csrValB, csrRowPtrC, csrRowPtrCt_d, &csrColIndCt, &csrValCt, n, nnzCt, m, queue_one.data(), control);
    // STAGE 3 - STEP 2 : malloc C on devices
    run_status = clEnqueueReadBuffer(control->queue(),
                                     csrRowPtrC,
                                     1,
                                     0,
                                     (m + 1)*sizeof(cl_int),
                                     csrRowPtrC_h.data(),
                                     0,
                                     0,
                                     0);

    int old_val, new_val;
    old_val = csrRowPtrC_h[0];
    csrRowPtrC_h[0] = 0;
    for (int i = 1; i <= m; i++)
    {
        new_val = csrRowPtrC_h[i];
        csrRowPtrC_h[i] = old_val + csrRowPtrC_h[i-1];
        old_val = new_val;
        //cout <<  csrRowPtrC_h[i] << " ";
    }
    //cout << endl;

    int nnzC = csrRowPtrC_h[m];
    //std::cout << "nnzC = " << nnzC << std::endl;
    
    matC->colIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzC * sizeof( cl_int ), NULL, &run_status );
    matC->values =     ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE, nnzC * sizeof( cl_float ), NULL, &run_status );
    
    cl_mem csrColIndC = matC->colIndices;
    cl_mem csrValC    = matC->values;

    run_status = clEnqueueWriteBuffer(control->queue(),
                                     csrRowPtrC,
                                     1,
                                     0,
                                     (m + 1)*sizeof(cl_int),
                                     csrRowPtrC_h.data(),
                                     0,
                                     0,
                                     0);


    copy_Ct_to_C_opencl(counter_one.data(), csrValC, csrRowPtrC, csrColIndC, csrValCt, csrRowPtrCt_d, csrColIndCt, queue_one_d, control);
    
    matC->num_rows = m;
    matC->num_cols = n;
    matC->num_nonzeros  = nnzC;
 
    ::clReleaseMemObject(csrRowPtrCt_d);
    ::clReleaseMemObject(queue_one_d);
    ::clReleaseMemObject(csrColIndCt);
    ::clReleaseMemObject(csrValCt);
}

int statistics(int *_h_csrRowPtrCt, int *_h_counter, int *_h_counter_one, int *_h_counter_sum, int *_h_queue_one, int _m)
{
    int nnzCt = 0;
    int _nnzCt_full = 0;

    // statistics for queues
    int count, position;

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            _h_counter_one[count]++;
            _h_counter_sum[count] += count;
            _nnzCt_full += count;
        }
        else if (count >= 122 && count <= 128)
        {
            _h_counter_one[122]++;
            _h_counter_sum[122] += count;
            _nnzCt_full += count;
        }
        else if (count >= 129 && count <= 256)
        {
            _h_counter_one[123]++;
            _h_counter_sum[123] += count;
            _nnzCt_full += count;
        }
        else if (count >= 257 && count <= 512)
        {
            _h_counter_one[124]++;
            _h_counter_sum[124] += count;
            _nnzCt_full += count;
        }
        else if (count >= 513)
        {
            _h_counter_one[127]++;
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _nnzCt_full += count;
        }
    }

    // exclusive scan

    int old_val, new_val;

    old_val = _h_counter_one[0];
    _h_counter_one[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_one[i];
        _h_counter_one[i] = old_val + _h_counter_one[i-1];
        old_val = new_val;
    }

    old_val = _h_counter_sum[0];
    _h_counter_sum[0] = 0;
    for (int i = 1; i <= NUM_SEGMENTS; i++)
    {
        new_val = _h_counter_sum[i];
        _h_counter_sum[i] = old_val + _h_counter_sum[i-1];
        old_val = new_val;
    }

    nnzCt = _h_counter_sum[NUM_SEGMENTS];

    for (int i = 0; i < _m; i++)
    {
        count = _h_csrRowPtrCt[i];

        if (count >= 0 && count <= 121)
        {
            position = _h_counter_one[count] + _h_counter[count];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[count];
            _h_counter_sum[count] += count;
            _h_counter[count]++;
        }
        else if (count >= 122 && count <= 128)
        {
            position = _h_counter_one[122] + _h_counter[122];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[122];
            _h_counter_sum[122] += count;
            _h_counter[122]++;
        }
        else if (count >= 129 && count <= 256)
        {
            position = _h_counter_one[123] + _h_counter[123];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[123];
            _h_counter_sum[123] += count;
            _h_counter[123]++;
        }
        else if (count >= 257 && count <= 512)
        {
            position = _h_counter_one[124] + _h_counter[124];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[124];
            _h_counter_sum[124] += count;
            _h_counter[124]++;
        }
        else if (count >= 513)
        {
            position = _h_counter_one[127] + _h_counter[127];
            _h_queue_one[TUPLE_QUEUE * position] = i;
            _h_queue_one[TUPLE_QUEUE * position + 1] = _h_counter_sum[127];
            _h_counter_sum[127] += MERGELIST_INITSIZE;
            _h_counter[127]++;
        }
    }

    return nnzCt;
}

