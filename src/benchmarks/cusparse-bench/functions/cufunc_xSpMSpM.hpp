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

#ifndef CUSPARSE_BENCHMARK_xSpMSpM_HXX__
#define CUSPARSE_BENCHMARK_xSpMSpM_HXX__

#include "cufunc_common.hpp"
#include "include/mm_reader.hpp"
#include "include/io-exception.hpp"

// C = alpha * A * A
template<typename T>
class xSpMSpM : public cusparseFunc {
public:
    xSpMSpM(StatisticalTimer& timer) : cusparseFunc(timer)
    {
        alpha = 1.0;
        beta  = 1.0;

        n_rows = 0;
        n_cols = 0;
        n_vals = 0;

        dev_csrValA = nullptr;
        dev_csrRowPtrA = nullptr;
        dev_csrColIndA = nullptr;

        dev_csrValC    = nullptr;
        dev_csrRowPtrC = nullptr;
        dev_csrColIndC = nullptr;

        buffer = nullptr;

        cusparseStatus_t err = cusparseCreateMatDescr(&descrA);
        CUDA_V_THROW(err, "cusparseCreateMatDescr  A failed");

        err = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUDA_V_THROW(err, "cusparseSetMatType A failed");

        err = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        CUDA_V_THROW(err, "cusparseSetMatIndexBase  A failed");

        // D Matrix is not used, only for sake of arguments it is created
        err = cusparseCreateMatDescr(&descrD);
        CUDA_V_THROW(err, "cusparseCreateMatDescr  D failed");

        err = cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUDA_V_THROW(err, "cusparseSetMatType D failed");

        err = cusparseSetMatIndexBase(descrD, CUSPARSE_INDEX_BASE_ZERO);
        CUDA_V_THROW(err, "cusparseSetMatIndexBase  D failed");

        err = cusparseCreateMatDescr(&descrC);
        CUDA_V_THROW(err, "cusparseCreateMatDescr C failed");

        err = cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUDA_V_THROW(err, "cusparseSetMatType C failed");

        err = cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
        CUDA_V_THROW(err, "cusparseSetMatIndexBase C failed");

    }// end of C'tor

    ~xSpMSpM()
    {
        cusparseDestroyMatDescr(descrA);
        cusparseDestroyMatDescr(descrD);
        cusparseDestroyMatDescr(descrC);
    }// end of D'tor

    void call_func()
    {
        timer.Start(timer_id);
        xSpMSpM_Function(true);
        timer.Stop(timer_id);
    }

    double gflops()
    {
        return 0.0;
    }

    std::string gflops_formula()
    {
        return "N/A";
    }

    double bandwidth()
    {
        //  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return (sizeof(int)*(n_vals + n_rows) + sizeof(T) * (n_vals + n_cols + n_rows)) / time_in_ns();
    }

    std::string bandwidth_formula()
    {
        return "GiB/s";
    }

    void setup_buffer(double alpha, double beta, const std::string& path)
    {
        initialize_scalars(alpha, beta);

        if (sparseHeaderfromFile(&n_vals, &n_rows, &n_cols, path.c_str()))
        {
            throw clsparse::io_exception("Could not read matrix market header from disk");
        }

        if (csrMatrixfromFile(row_offsets, col_indices, values, path.c_str()))
        {
            throw clsparse::io_exception("Could not read matrix market header from disk");
        }

        cudaError_t err = cudaMalloc((void**)&dev_csrRowPtrA, row_offsets.size() * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");

        err = cudaMalloc((void**)&dev_csrColIndA, col_indices.size() * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc device_col_indices");

        err = cudaMalloc((void**)&dev_csrValA, values.size() * sizeof(T));
        CUDA_V_THROW(err, "cudaMalloc device_values");
        
    }// end of function

    void initialize_cpu_buffer()
    {

    }

    void initialize_gpu_buffer()
    {
        cudaError_t err = cudaMemcpy(dev_csrRowPtrA, &row_offsets[0], row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");

        err = cudaMemcpy(dev_csrColIndA, &col_indices[0], col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");

        err = cudaMemcpy(dev_csrValA, &values[0], values.size() * sizeof(T), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");
#if 0
        nnzTotalDevHostPtr = &nnzC;
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

        //  step 1: create an opaque structure
        cusparseCreateCsrgemm2Info(&info);

        // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
        createBuffersNNZ_C();
#endif
    }// end of function

    void reset_gpu_write_buffer()
    {
#if 0
        cudaError_t err = cudaMemset(dev_csrValC, 0x0, nnzC * sizeof(T));
        CUDA_V_THROW(err, "cudaMemset dev_csrValC " + std::to_string(nnzC));

        err = cudaMemset(dev_csrColIndC, 0x0, nnzC * sizeof(int));
        CUDA_V_THROW(err, "cudaMemset dev_csrColIndC " + std::to_string(nnzC));

        err = cudaMemset(dev_csrRowPtrC, 0x0, (n_rows+1) * sizeof(int));
        CUDA_V_THROW(err, "cudaMemset dev_csrRowPtrC " + std::to_string(nnzC));
#endif

        CUDA_V_THROW(cudaFree(dev_csrValC), "cudafree dev_csrValC");
        CUDA_V_THROW(cudaFree(dev_csrRowPtrC), "cudafree dev_csrRowPtrC");
        CUDA_V_THROW(cudaFree(dev_csrColIndC), "cudafree dev_csrColIndC");

        CUDA_V_THROW(cudaFree(buffer), "cudafree buffer");

        // step 5: destroy the opaque structure
        cusparseDestroyCsrgemm2Info(info);
    }

    void read_gpu_buffer()
    {
    }

    void releaseGPUBuffer_deleteCPUBuffer()
    {
        CUDA_V_THROW(cudaFree(dev_csrValA),    "cudafree dev_csrValA");
        CUDA_V_THROW(cudaFree(dev_csrRowPtrA), "cudafree dev_csrRowPtrA");
        CUDA_V_THROW(cudaFree(dev_csrColIndA), "cudafree dev_csrColIndA");
#if 0
        CUDA_V_THROW(cudaFree(dev_csrValC),    "cudafree dev_csrValC");
        CUDA_V_THROW(cudaFree(dev_csrRowPtrC), "cudafree dev_csrRowPtrC");
        CUDA_V_THROW(cudaFree(dev_csrColIndC), "cudafree dev_csrColIndC");

        CUDA_V_THROW(cudaFree(buffer), "cudafree buffer");

        // step 5: destroy the opaque structure
        cusparseDestroyCsrgemm2Info(info);
#endif

        row_offsets.clear();
        col_indices.clear();
        values.clear();
    }

protected:
    void initialize_scalars(double pAlpha, double pBeta)
    {
        alpha = makeScalar< T >(pAlpha);
        beta  = makeScalar< T >(pBeta);

        beta = 0.0; // C = alpha* A * A + beta * D; 
    }

private:
    void createBuffersNNZ_C(void);
    void xSpMSpM_Function(bool flush);

    //Input host matrix in csr format : A
    std::vector< int > row_offsets;
    std::vector< int > col_indices;
    std::vector< T > values;

    T alpha;
    T beta;
    int n_rows;
    int n_cols;
    int n_vals; 

    csrgemm2Info_t info;
    cusparseMatDescr_t descrA;
    cusparseMatDescr_t descrD;
    cusparseMatDescr_t descrC;

    // device CUDA pointers
    T*   dev_csrValA;
    int* dev_csrRowPtrA;
    int* dev_csrColIndA;

    T*   dev_csrValC;
    int* dev_csrRowPtrC;
    int* dev_csrColIndC;

    int* nnzTotalDevHostPtr; // Points to host memory
    int baseC;
    int nnzC;
    void* buffer;
    size_t bufferSize;
};


template<> void
xSpMSpM<float> ::createBuffersNNZ_C(void)
{
    double betaT = 0.0;
    size_t nnzA = values.size();

    // Step 2: allocate buffer for csrgemm2Nnzand csrgemm2
    cuSparseStatus =  cusparseScsrgemm2_bufferSizeExt(handle, n_rows, n_cols, n_cols, &alpha,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA, nullptr, // beta is nullptr => C = alpha*A*B
        descrD, 0, nullptr, nullptr,
        info,
        &bufferSize);

    CUDA_V_THROW(cuSparseStatus, "cusparseScsrgemm2_bufferSizeExt() failed \n");

    cudaError_t err = cudaMalloc(&buffer, bufferSize);
    CUDA_V_THROW(err, "cudaMalloc buffer in createBuffersNNZ_C");

    // step 3: compute dev_csrRowPtrC
    err = cudaMalloc((void**)&dev_csrRowPtrC, sizeof(int) * (n_rows + 1));
    CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

    cuSparseStatus = cusparseXcsrgemm2Nnz(handle, n_rows, n_cols, n_cols,
                         descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA, 
                         descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA, 
                         descrD, 0, nullptr, nullptr, 
                         descrC, dev_csrRowPtrC, nnzTotalDevHostPtr, info, buffer);
    
    CUDA_V_THROW(cuSparseStatus, "cusparseXcsrgemm2Nnz() failed \n");

    cudaDeviceSynchronize(); // Check this !!!!

    if (NULL != nnzTotalDevHostPtr)
    {
        nnzC = *nnzTotalDevHostPtr;
    }
    else
    { 
        cudaMemcpy(&nnzC,  dev_csrRowPtrC + n_rows, sizeof(int), cudaMemcpyDeviceToHost); 
        cudaMemcpy(&baseC, dev_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost); 
        nnzC -= baseC; 
    }
    
   err = cudaMalloc((void**)&dev_csrColIndC, sizeof(int)*nnzC);
   CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

   err = cudaMalloc((void**)&dev_csrValC, sizeof(float)*nnzC);
   CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

}// end of function


template<> void
xSpMSpM<double> ::createBuffersNNZ_C(void)
{
    size_t nnzA = values.size();

    // Step 2: allocate buffer for csrgemm2Nnzand csrgemm2
    cuSparseStatus = cusparseDcsrgemm2_bufferSizeExt(handle, n_rows, n_cols, n_cols, &alpha,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA, nullptr, // beta is nullptr => C = alpha*A*B
        descrD, 0, nullptr, nullptr,
        info,
        &bufferSize);

    CUDA_V_THROW(cuSparseStatus, "cusparseScsrgemm2_bufferSizeExt() failed \n");

    cudaError_t err = cudaMalloc(&buffer, bufferSize);
    CUDA_V_THROW(err, "cudaMalloc buffer in createBuffersNNZ_C");

    // step 3: compute dev_csrRowPtrC
    err = cudaMalloc((void**)&dev_csrRowPtrC, sizeof(int) * (n_rows + 1));
    CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

    cuSparseStatus = cusparseXcsrgemm2Nnz(handle, n_rows, n_cols, n_cols,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA,
        descrA, nnzA, dev_csrRowPtrA, dev_csrColIndA,
        descrD, 0, nullptr, nullptr,
        descrC, dev_csrRowPtrC, nnzTotalDevHostPtr, info, buffer);

    CUDA_V_THROW(cuSparseStatus, "cusparseXcsrgemm2Nnz() failed \n");

    cudaDeviceSynchronize(); // Check this !!!!

    if (NULL != nnzTotalDevHostPtr)
    {
        nnzC = *nnzTotalDevHostPtr;
    }
    else
    {
        cudaMemcpy(&nnzC, dev_csrRowPtrC + n_rows, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, dev_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    err = cudaMalloc((void**)&dev_csrColIndC, sizeof(int)*nnzC);
    CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

    err = cudaMalloc((void**)&dev_csrValC, sizeof(double)*nnzC);
    CUDA_V_THROW(err, "cudaMalloc dev_csrRowPtrC failed in createBuffersNNZ_C");

}// end of function


template<> void
xSpMSpM<float> ::xSpMSpM_Function(bool flush)
{
    nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    //  step 1: create an opaque structure
    cusparseCreateCsrgemm2Info(&info);

    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    createBuffersNNZ_C();

    size_t nnzA = values.size();
    // step 4: finish sparsity pattern and value of C 
    // Remark: set csrValC to null if only sparsity pattern is required. 
    cuSparseStatus = cusparseScsrgemm2(handle, n_rows, n_cols, n_cols, &alpha,
                                       descrA, nnzA, dev_csrValA, dev_csrRowPtrA, dev_csrColIndA,
                                       descrA, nnzA, dev_csrValA, dev_csrRowPtrA, dev_csrColIndA,
                                       nullptr, descrD, 0, nullptr, nullptr, nullptr, 
                                       descrC, dev_csrValC, dev_csrRowPtrC, dev_csrColIndC, 
                                       info, buffer); 

    cudaDeviceSynchronize();

}// end of function



template<> void
xSpMSpM<double> ::xSpMSpM_Function(bool flush)
{
    nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    //  step 1: create an opaque structure
    cusparseCreateCsrgemm2Info(&info);

    // step 2: allocate buffer for csrgemm2Nnz and csrgemm2
    createBuffersNNZ_C();

    size_t nnzA = values.size();
    // step 4: finish sparsity pattern and value of C 
    // Remark: set csrValC to null if only sparsity pattern is required. 
    cuSparseStatus = cusparseDcsrgemm2(handle, n_rows, n_cols, n_cols, &alpha,
                                       descrA, nnzA, dev_csrValA, dev_csrRowPtrA, dev_csrColIndA,
                                       descrA, nnzA, dev_csrValA, dev_csrRowPtrA, dev_csrColIndA,
                                       nullptr, descrD, 0, nullptr, nullptr, nullptr,
                                       descrC, dev_csrValC, dev_csrRowPtrC, dev_csrColIndC, 
                                       info, buffer); 

    cudaDeviceSynchronize();

}// end of function


#endif //CUSPARSE_BENCHMARK_xSpMSpM_HXX__