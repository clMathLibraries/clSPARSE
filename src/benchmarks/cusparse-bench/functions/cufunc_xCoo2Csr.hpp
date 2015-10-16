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

#ifndef CUBLAS_BENCHMARK_xCoo2Csr_HXX__
#define CUBLAS_BENCHMARK_xCoo2Csr_HXX__

#include "cufunc_common.hpp"
#include "include/io-exception.hpp"

template <typename T>
class xCoo2Csr: public cusparseFunc
{
public:
    xCoo2Csr( StatisticalTimer& timer ): cusparseFunc( timer )
    {
        n_rows = 0;
        n_cols = 0;
        n_vals = 0; // nnz
    }

    ~xCoo2Csr( )
    {

    }

    void call_func( )
    {
        timer.Start( timer_id );

        xCoo2Csr_Function( true );

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
        // Number of Elements converted in unit time
        return ( n_vals / time_in_ns( ) );
    }

    std::string bandwidth_formula( )
    {
        return "GiElements/s";
    }

    void setup_buffer( double alpha, double beta, const std::string& path )
    {
        int fileError = sparseHeaderfromFile( &n_vals, &n_rows, &n_cols, path.c_str( ) );
        if( fileError != 0 )
        {
            throw clsparse::io_exception( "Could not read matrix market header from disk" + path);
        }

        if( cooMatrixfromFile( row_indices, col_indices, values, path.c_str( ) ) )
        {
            throw clsparse::io_exception( "Could not read matrix market from disk: " + path );
        }

        // Input: COO Row Indices
        err = cudaMalloc( (void**)&deviceCooRowInd, n_vals * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc deviceCooRowInd" );

        // Output: CSR
        cudaError_t err = cudaMalloc( (void**)&deviceCSRRowOffsets, ( n_rows + 1 ) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc deviceCSRRowOffsets" );

    }// End of function

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        cudaError_t err = cudaMemcpy( deviceCooRowInd, &row_indices[ 0 ], row_indices.size( ) * sizeof( int ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc deviceCSRRowOffsets" );

        err = cudaMemset( deviceCSRRowOffsets, 0x0, ( n_rows + 1 ) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMemset deviceCSRRowOffsets" );

    }// end of function

    void reset_gpu_write_buffer( )
    {
        err = cudaMemset( deviceCSRRowOffsets, 0x0, ( n_rows + 1 ) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMemset deviceCSRRowOffsets" );

        cudaDeviceSynchronize( );
    }// end of function

    void read_gpu_buffer( )
    {
    }

    void releaseGPUBuffer_deleteCPUBuffer( )
    {
        CUDA_V_THROW( cudaFree( deviceCSRRowOffsets ), "cudafree deviceCSRRowOffsets" );
        CUDA_V_THROW( cudaFree( deviceCooRowInd ), "cudafree deviceCooRowInd" );

        row_indices.clear( );
        col_indices.clear( );
        values.clear( );
    }

protected:
    void initialize_scalars( double pAlpha, double pBeta )
    {
    }

private:
    void xCoo2Csr_Function( bool flush );

    //host matrix definition corresponding to CSR Format
    std::vector< int > row_indices;
    std::vector< int > col_indices;
    std::vector< T > values; // matrix values

    int  n_rows; // number of rows
    int  n_cols; // number of cols
    int  n_vals; // number of Non-Zero Values (nnz)
    int* colIndices;

    // device CUDA pointers
    int* deviceCSRRowOffsets; // Input: CSR Row Offsets
    int* deviceCooRowInd; // Output: Coordinate format row indices
}; // class xCoo2Csr

template<>
void
xCoo2Csr<float>::
xCoo2Csr_Function( bool flush )
{
    cuSparseStatus = cusparseXcoo2csr( handle,
                                       deviceCooRowInd,
                                       n_vals,
                                       n_rows,
                                       deviceCSRRowOffsets,
                                       CUSPARSE_INDEX_BASE_ZERO );

    CUDA_V_THROW( cuSparseStatus, "cusparseCoo2Csr" );

    cudaDeviceSynchronize( );

}


template<>
void
xCoo2Csr<double>::
xCoo2Csr_Function( bool flush )
{
    cuSparseStatus = cusparseXcoo2csr( handle,
                                       deviceCooRowInd,
                                       n_vals,
                                       n_rows,
                                       deviceCSRRowOffsets,
                                       CUSPARSE_INDEX_BASE_ZERO );

    CUDA_V_THROW( cuSparseStatus, "cusparseCoo2Csr" );

    cudaDeviceSynchronize( );

}

#endif //CUBLAS_BENCHMARK_xCoo2Csr_HXX__
