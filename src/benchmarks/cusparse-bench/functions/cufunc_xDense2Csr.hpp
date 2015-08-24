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

#ifndef CUBLAS_BENCHMARK_xDense2Csr_HXX__
#define CUBLAS_BENCHMARK_xDense2Csr_HXX__

#include "cufunc_common.hpp"
#include "include/io-exception.hpp"

template <typename T>
class xDense2Csr : public cusparseFunc
{
public:
    xDense2Csr(StatisticalTimer& timer) : cusparseFunc(timer)
    {
        cusparseStatus_t err = cusparseCreateMatDescr(&descrA);
        CUDA_V_THROW(err, "cusparseCreateMatDescr failed");

        err = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
        CUDA_V_THROW(err, "cusparseSetMatType failed");

        err = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
        CUDA_V_THROW(err, "cusparseSetMatIndexBase failed");

        n_rows = 0;
        n_cols = 0;
        n_vals = 0;

        device_col_indices = nullptr;
        device_row_offsets = nullptr;

        device_values = nullptr;
        device_A      = nullptr;
        nnzPerRow     = nullptr;

        devRowOffsets = nullptr;
        devColIndices = nullptr;
        devValues     = nullptr;
    }// end

    ~xDense2Csr()
    {
        cusparseDestroyMatDescr(descrA);
    }// end

    void call_func()
    {
        timer.Start(timer_id);
        xDense2Csr_Function(true);
        timer.Stop(timer_id);
    }// end

    double gflops()
    {
        return 0.0;
    } // end

    std::string gflops_formula()
    {
        return "N/A";
    } // end

    double bandwidth()
    {
        // Number of Elements processed in unit time
        return (n_rows * n_cols / time_in_ns());
    }

    std::string bandwidth_formula()
    {
        return "GiElements/s";
    } // end

    void setup_buffer(double alpha, double beta, const std::string& path)
    {
        int fileError = sparseHeaderfromFile(&n_vals, &n_rows, &n_cols, path.c_str());
        if (fileError != 0)
        {
            throw clsparse::io_exception( "Could not read matrix market header from disk: " + path);
        }

        if (csrMatrixfromFile(row_offsets, col_indices, values, path.c_str()))
        {
            throw clsparse::io_exception( "Could not read matrix market from disk: " + path);
        }

        cudaError_t err = cudaMalloc((void**)&device_row_offsets, (n_rows + 1) * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");

        err = cudaMalloc((void**)&device_col_indices, n_vals * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc device_col_indices");

        err = cudaMalloc((void**)&device_values, n_vals * sizeof(T));
        CUDA_V_THROW(err, "cudaMalloc device_values");

        err = cudaMalloc((void**)&device_A, n_rows * n_cols * sizeof(T));
        CUDA_V_THROW(err, "cudaMalloc device_A");

        // Output CSR
        err = cudaMalloc((void**)&devRowOffsets, (n_rows + 1) * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc devRowOffsets");

        err = cudaMalloc((void**)&devColIndices, n_vals * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc devColIndices");

        err = cudaMalloc((void**)&devValues, n_vals * sizeof(T));
        CUDA_V_THROW(err, "cudaMalloc devValues");

        // Allocate memory for nnzPerRow
        err = cudaMalloc((void**)&nnzPerRow, n_rows * sizeof(int));
        CUDA_V_THROW(err, "cudaMalloc nnzPerRow");

    }// end

    void initialize_cpu_buffer()
    {
    }

    void initialize_gpu_buffer()
    {
        cudaError_t err = cudaMemcpy(device_row_offsets, &row_offsets[0], row_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_row_offsets");

        err = cudaMemcpy(device_col_indices, &col_indices[0], col_indices.size() * sizeof(int), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_col_indices");

        err = cudaMemcpy(device_values, &values[0], values.size() * sizeof(T), cudaMemcpyHostToDevice);
        CUDA_V_THROW(err, "cudaMalloc device_values");

        err = cudaMemset(device_A, 0x0, n_rows * n_cols * sizeof(T));
        CUDA_V_THROW(err, "cudaMalloc device_A");

        // call csr2dense to get input in dense format
        csr2dense_Function(true);

        int nnzA;
        // Compute number of nonzero elements per row
        if (typeid(T) == typeid(float))
        {
            cuSparseStatus = cusparseSnnz(handle,
                CUSPARSE_DIRECTION_ROW,
                n_rows,
                n_cols,
                descrA,
                reinterpret_cast< float*> (device_A),
                n_rows,
                nnzPerRow,
                &nnzA);
            CUDA_V_THROW(cuSparseStatus, "cusparseSnnz");
        }
        else if (typeid(T) == typeid(double))
        {
            cuSparseStatus = cusparseDnnz(handle,
                CUSPARSE_DIRECTION_ROW,
                n_rows,
                n_cols,
                descrA,
               reinterpret_cast< double*> (device_A),
                n_rows,
                nnzPerRow,
                &nnzA);
            CUDA_V_THROW(cuSparseStatus, "cusparseDnnz");
        }
        else
        {
            // error
        }

        if (nnzA != n_vals)
        {
            // error
        }
        cudaDeviceSynchronize();
        // Once we get input in dense format, no-loner input csr values are needed.
        CUDA_V_THROW(cudaFree(device_values), "cudafree device_values");
        CUDA_V_THROW(cudaFree(device_row_offsets), "cudafree device_row_offsets");
        CUDA_V_THROW(cudaFree(device_col_indices), "cudafree device_col_indices");

    }// end

    void reset_gpu_write_buffer()
    {
        cudaError_t err = cudaMemset(devRowOffsets, 0x0, (n_rows + 1) * sizeof(int));
        CUDA_V_THROW(err, "cudaMemset reset_gpu_write_buffer: devRowOffsets");

        err = cudaMemset(devColIndices, 0x0, n_vals * sizeof(int));
        CUDA_V_THROW(err, "cudaMemset reset_gpu_write_buffer: devColIndices");

        err = cudaMemset(devValues, 0x0, n_vals * sizeof(T));
        CUDA_V_THROW(err, "cudaMemset reset_gpu_write_buffer: devValues");
    }

    void read_gpu_buffer()
    {
    }

    void releaseGPUBuffer_deleteCPUBuffer()
    {
        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor

        CUDA_V_THROW(cudaFree(device_A), "cudafree device_A");

        CUDA_V_THROW(cudaFree(devValues), "cudafree devValues");
        CUDA_V_THROW(cudaFree(devRowOffsets), "cudafree devRowOffsets");
        CUDA_V_THROW(cudaFree(devColIndices), "cudafree devColIndices");
        CUDA_V_THROW(cudaFree(nnzPerRow), "cudafree nnzPerRow");

        row_offsets.clear();
        col_indices.clear();
        values.clear();
    }//end

protected:
    void initialize_scalars(double pAlpha, double pBeta)
    {
    }

private:
    void xDense2Csr_Function(bool flush);
    void csr2dense_Function(bool flush); // to get input in dense format

    //host matrix definition in csr format
    std::vector< int > row_offsets;
    std::vector< int > col_indices;
    std::vector< T > values;

    int  n_rows; // number of rows
    int  n_cols; // number of cols
    int  n_vals; // number of Non-Zero Values (nnz)

    cusparseMatDescr_t descrA;

    // device CUDA pointers
    int* device_row_offsets;
    int* device_col_indices;
    T* device_values;
    // Dense format: output - >input
    T* device_A;
    int* nnzPerRow; // Number of non-zero elements per row

    // Output devie CUDA pointers:csr format
    int* devRowOffsets;
    int* devColIndices;
    T* devValues;


}; // xDense2Csr

template<>
void
xDense2Csr<float>::
csr2dense_Function(bool flush)
{
    cuSparseStatus = cusparseScsr2dense(handle,
                                         n_rows,
                                         n_cols,
                                         descrA,
                                         device_values,
                                         device_row_offsets,
                                         device_col_indices,
                                         device_A,
                                         n_rows);  //dense Matrix A  stored in Col-major format
    CUDA_V_THROW(cuSparseStatus, "cusparseScsr2dense");

    cudaDeviceSynchronize();
}// end

template<>
void
xDense2Csr<double>::
csr2dense_Function(bool flush)
{
    cuSparseStatus = cusparseDcsr2dense(handle,
                                         n_rows,
                                         n_cols,
                                         descrA,
                                         device_values,
                                         device_row_offsets,
                                         device_col_indices,
                                         device_A,
                                         n_rows); //dense Matrix A  stored in Col-major format
    CUDA_V_THROW(cuSparseStatus, "cusparseDcsr2dense");

    cudaDeviceSynchronize();
}// end of function


template<>
void
xDense2Csr<float>::
xDense2Csr_Function(bool flush)
{
    cuSparseStatus = cusparseSdense2csr(handle,
        n_rows,
        n_cols,
        descrA,
        device_A,
        n_rows,    // dense matrix in col-major format, lda is number of elements in major dimension (number of rows)
        nnzPerRow,
        devValues,
        devRowOffsets,
        devColIndices);

    CUDA_V_THROW(cuSparseStatus, "cusparseSdense2csr");
    cudaDeviceSynchronize();
} // end of function


template<>
void
xDense2Csr<double>::
xDense2Csr_Function(bool flush)
{
    cuSparseStatus = cusparseDdense2csr(handle,
        n_rows,
        n_cols,
        descrA,
        device_A,
        n_rows,  // dense matrix in col-major format, lda is number of elements in major dimension (number of rows)
        nnzPerRow,
        devValues,
        devRowOffsets,
        devColIndices);

    CUDA_V_THROW(cuSparseStatus, "cusparseDdense2csr");
    cudaDeviceSynchronize();
}// end of function



#endif // CUBLAS_BENCHMARK_xDense2Csr_HXX__
