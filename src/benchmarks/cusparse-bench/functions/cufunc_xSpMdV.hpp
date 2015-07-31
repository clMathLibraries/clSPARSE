/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/

#ifndef CUBLAS_BENCHMARK_xSpMdV_HXX__
#define CUBLAS_BENCHMARK_xSpMdV_HXX__

#include "cufunc_common.hpp"
#include "include/mm_reader.hpp"
#include "include/io-exception.hpp"

template <typename T>
class xSpMdV : public cusparseFunc
{
public:
    xSpMdV( StatisticalTimer& timer ): cusparseFunc( timer ), transA( CUSPARSE_OPERATION_NON_TRANSPOSE )
    {
        cusparseStatus_t err = cusparseCreateMatDescr( &descrA );
        CUDA_V_THROW( err, "cusparseCreateMatDescr failed" );

        err = cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
        CUDA_V_THROW( err, "cusparseSetMatType failed" );

        err = cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO );
        CUDA_V_THROW( err, "cusparseSetMatIndexBase failed" );
    }

    ~xSpMdV( )
    {
        cusparseDestroyMatDescr( descrA );
    }

    void call_func( )
    {
        timer.Start( timer_id );
        xSpMdV_Function( true );
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
        //  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return ( sizeof( int )*( n_vals + n_rows ) + sizeof( T ) * ( n_vals + n_cols + n_rows ) ) / time_in_ns( );
    }

    std::string bandwidth_formula( )
    {
        return "GiB/s";
    }

    void setup_buffer( double alpha, double beta, const std::string& path )
    {
        initialize_scalars( alpha, beta );

        if( sparseHeaderfromFile( &n_vals, &n_rows, &n_cols, path.c_str( ) ) )
        {
            throw clsparse::io_exception( "Could not read matrix market header from disk" );
        }

        if (csrMatrixfromFile( row_offsets, col_indices, values, path.c_str( ) ) )
        {
            throw clsparse::io_exception( "Could not read matrix market header from disk" );
        }

        cudaError_t err = cudaMalloc( (void**) &device_row_offsets, row_offsets.size( ) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMalloc( (void**) &device_col_indices, col_indices.size( ) * sizeof( int ) );
        CUDA_V_THROW( err, "cudaMalloc device_col_indices" );

        err = cudaMalloc( (void**) &device_values, values.size( ) * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_values" );

        err = cudaMalloc( (void**) &device_x, n_cols * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_values" );

        err = cudaMalloc( (void**) &device_y, n_rows * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_values" );
    }

    void initialize_cpu_buffer( )
    {
        // Initialize x vector to size n_cols filled with all 1's
        x = std::vector< T >( n_cols, 1 );
    }

    void initialize_gpu_buffer( )
    {
        cudaError_t err = cudaMemcpy( device_row_offsets, &row_offsets[ 0 ], row_offsets.size( ) * sizeof( int ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMemcpy( device_col_indices, &col_indices[ 0 ], col_indices.size( ) * sizeof( int ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMemcpy( device_values, &values[ 0 ], values.size( ) * sizeof( T ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMemcpy( device_x, &x[ 0 ], x.size( ) * sizeof( T ), cudaMemcpyHostToDevice );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );

        err = cudaMemset( device_y, 0x0, n_rows * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMalloc device_row_offsets" );
    }

    void reset_gpu_write_buffer( )
    {
        cudaError_t err = cudaMemset( device_y, 0x0, n_rows * sizeof( T ) );
        CUDA_V_THROW( err, "cudaMemset device_y " + std::to_string(n_rows)  );
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
        CUDA_V_THROW( cudaFree( device_x ), "cudafree device_x" );
        CUDA_V_THROW( cudaFree( device_y ), "cudafree device_y" );

        row_offsets.clear( );
        col_indices.clear( );
        values.clear( );
        x.clear( );
    }

protected:
    void initialize_scalars( double pAlpha, double pBeta )
    {
        alpha = makeScalar< T >( pAlpha );
        beta = makeScalar< T >( pBeta );
    }

private:
    void xSpMdV_Function( bool flush );

    //host matrix definition
    std::vector< int > row_offsets;
    std::vector< int > col_indices;
    std::vector< T > values;
    std::vector< T > x;
    int n_rows;
    int n_cols;
    int n_vals;

    T alpha;
    T beta;
    cusparseOperation_t transA;
    cusparseMatDescr_t descrA;

    // device CUDA pointers
    int* device_row_offsets;
    int* device_col_indices;
    T* device_values;
    T* device_x;
    T* device_y;

}; // class xSpMdV

template<>
void 
xSpMdV<float>::
xSpMdV_Function( bool flush )
{
    cuSparseStatus = cusparseScsrmv( handle,
                               transA,
                               n_rows,
                               n_cols,
                               n_vals,
                               &alpha,
                               descrA,
                               device_values,
                               device_row_offsets,
                               device_col_indices,
                               device_x,
                               &beta,
                               device_y );

    cudaDeviceSynchronize( );
}

template<>
void 
xSpMdV<double>::
xSpMdV_Function( bool flush )
{
    cuSparseStatus = cusparseDcsrmv( handle,
                                transA,
                                n_rows,
                                n_cols,
                                n_vals,
                                &alpha,
                                descrA,
                                device_values,
                                device_row_offsets,
                                device_col_indices,
                                device_x,
                                &beta,
                                device_y );

    cudaDeviceSynchronize( );
}

#endif // ifndef CUBLAS_BENCHMARK_xSpMdV_HXX__
