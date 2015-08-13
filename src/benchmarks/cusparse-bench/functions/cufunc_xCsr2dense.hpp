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

#ifndef CUBLAS_BENCHMARK_xCsr2Dense_HXX__
#define CUBLAS_BENCHMARK_xCsr2Dense_HXX__

#include "cufunc_common.hpp"
#include "include/io-exception.hpp"

template <typename T>
class xCsr2Dense : public cusparseFunc
{
public:
    xCsr2Dense( StatisticalTimer& timer ): cusparseFunc( timer )
    {
        cusparseStatus_t err = cusparseCreateMatDescr( &descrA );
        CUDA_V_THROW( err, "cusparseCreateMatDescr failed" );

        err = cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
        CUDA_V_THROW( err, "cusparseSetMatType failed" );

        err = cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO );
        CUDA_V_THROW( err, "cusparseSetMatIndexBase failed" );

        n_rows = 0;
        n_cols = 0;
        n_vals = 0;
    }

    ~xCsr2Dense( )
    {
        cusparseDestroyMatDescr( descrA );
    }

    void call_func( )
    {
        timer.Start( timer_id );
        xCsr2Dense_Function( true );
        timer.Stop( timer_id );
    }

    double gflops( )
    {
        return 0.0;
    }

    std::string gflops_formula( )
    {
        return "N/A";
    }

    double bandwidth( )
    {
#if 0
        //  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return ( sizeof( int )*( n_vals + n_rows ) + sizeof( T ) * ( n_vals + n_cols + n_rows ) ) / time_in_ns( );
#endif
		// Number of Elements converted in unit time
		return (n_vals / time_in_ns());
    }

    std::string bandwidth_formula( )
    {
        //return "GiB/s";
		return "GiElements/s";
    }

    void setup_buffer( double alpha, double beta, const std::string& path )
    {
        int fileError = sparseHeaderfromFile(&n_vals, &n_rows, &n_cols, path.c_str());
        if (fileError != 0)
        {
            throw clsparse::io_exception("Could not read matrix market header from disk");
        }

        if (csrMatrixfromFile( row_offsets, col_indices, values, path.c_str( ) ) )
        {
            throw clsparse::io_exception( "Could not read matrix market header from disk" );
        }

        //n_rows = row_offsets.size( );
        //n_cols = col_indices.size( );
        //n_vals = values.size( );

        cudaError_t err = cudaMalloc( (void**) &device_row_offsets, (n_rows + 1) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMalloc( (void**) &device_col_indices, n_vals * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc device_col_indices" );

        err = cudaMalloc( (void**) &device_values, n_vals * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_values" );

        err = cudaMalloc( (void**) &device_A, n_rows * n_cols * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_A" );
    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        cudaError_t err = cudaMemcpy( device_row_offsets, &row_offsets[ 0 ], row_offsets.size( ) * sizeof( int ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMemcpy( device_col_indices, &col_indices[ 0 ], col_indices.size( ) * sizeof( int ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_col_indices" );

        err = cudaMemcpy( device_values, &values[ 0 ], values.size( ) * sizeof( T ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_values" );

        err = cudaMemset( device_A, 0x0, n_rows * n_cols * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_A" );
    }

    void reset_gpu_write_buffer( )
    {
        cudaError_t err = cudaMemset( device_A, 0x0, n_rows * n_cols * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMemset reset_gpu_write_buffer" );

    }

    void read_gpu_buffer( )
    {
    }

    void releaseGPUBuffer_deleteCPUBuffer( )
    {
        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CUDA_V_THROW( cudaFree( device_values  ), "cudafree device_values" );
        CUDA_V_THROW( cudaFree( device_row_offsets ), "cudafree device_row_offsets" );
        CUDA_V_THROW( cudaFree( device_col_indices ), "cudafree device_col_indices" );
        CUDA_V_THROW( cudaFree( device_A ), "cudafree device_A" );

        row_offsets.clear( );
        col_indices.clear( );
        values.clear( );
    }

protected:
    void initialize_scalars( double pAlpha, double pBeta )
    {
    }

private:
    void xCsr2Dense_Function( bool flush );

    //host matrix definition
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
    T* device_A;

}; // class xCsr2Dense

template<>
void
xCsr2Dense<float>::
xCsr2Dense_Function( bool flush )
{
    cuSparseStatus =  cusparseScsr2dense( handle,
                                          n_rows,
                                          n_cols,
                                          descrA,
                                          device_values,
                                          device_row_offsets,
                                          device_col_indices,
                                          device_A,
                                          n_rows );
    CUDA_V_THROW( cuSparseStatus, "cusparseScsr2dense" );

    cudaDeviceSynchronize( );
}

template<>
void
xCsr2Dense<double>::
xCsr2Dense_Function( bool flush )
{
    cuSparseStatus = cusparseDcsr2dense( handle,
                                         n_rows,
                                         n_cols,
                                         descrA,
                                         device_values,
                                         device_row_offsets,
                                         device_col_indices,
                                         device_A,
                                         n_rows );
    CUDA_V_THROW( cuSparseStatus, "cusparseDcsr2dense" );

    cudaDeviceSynchronize( );
}

#endif // ifndef CUBLAS_BENCHMARK_xCsr2Dense_HXX__
