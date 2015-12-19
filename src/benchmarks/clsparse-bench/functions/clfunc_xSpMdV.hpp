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

#pragma once
#ifndef CLSPARSE_BENCHMARK_SPMV_HXX__
#define CLSPARSE_BENCHMARK_SPMV_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xSpMdV: public clsparseFunc
{
public:
    xSpMdV( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_bool extended_precision, cl_device_type devType, cl_bool keep_explicit_zeroes = true ): clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr )
    {
        //	Create and initialize our timer class, if the external timer shared library loaded
        if( sparseGetTimer )
        {
            gpuTimer = sparseGetTimer( CLSPARSE_GPU );
            gpuTimer->Reserve( 1, profileCount );
            gpuTimer->setNormalize( true );

            cpuTimer = sparseGetTimer( CLSPARSE_CPU );
            cpuTimer->Reserve( 1, profileCount );
            cpuTimer->setNormalize( true );

            gpuTimerID = gpuTimer->getUniqueID( "GPU xSpMdV", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xSpMdV", 0 );
        }

        clsparseEnableExtendedPrecision( control, extended_precision );
        explicit_zeroes = keep_explicit_zeroes;

        clsparseEnableAsync( control, false );
    }

    ~xSpMdV( )
    {
    }

    void call_func( )
    {
      if( gpuTimer && cpuTimer )
      {
        gpuTimer->Start( gpuTimerID );
        cpuTimer->Start( cpuTimerID );

        xSpMdV_Function( false );

        cpuTimer->Stop( cpuTimerID );
        gpuTimer->Stop( gpuTimerID );
      }
      else
      {
        xSpMdV_Function( false );
      }
    }

    double gflops( )
    {
        return ((2 * csrMtx.num_nonzeros) / time_in_ns ( ));
    }

    std::string gflops_formula( )
    {
        return "GFLOPs";
    }

    double bandwidth( )
    {
        //  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return (sizeof(clsparseIdx_t)*(csrMtx.num_nonzeros + csrMtx.num_rows) + sizeof(T) * (csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows)) / time_in_ns();
    }

    std::string bandwidth_formula( )
    {
        return "GiB/s";
    }


    void setup_buffer( double pAlpha, double pBeta, const std::string& path )
    {
        sparseFile = path;

        alpha = static_cast< T >( pAlpha );
        beta = static_cast< T >( pBeta );

        // Read sparse data from file and construct a COO matrix from it
        clsparseIdx_t nnz, row, col;
        clsparseStatus fileError = clsparseHeaderfromFile( &nnz, &row, &col, sparseFile.c_str( ) );
        if( fileError != clsparseSuccess )
            throw clsparse::io_exception( "Could not read matrix market header from disk: " + sparseFile );

        // Now initialize a CSR matrix from the COO matrix
        clsparseInitCsrMatrix( &csrMtx );
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows = row;
        csrMtx.num_cols = col;

        cl_int status;
        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.values" );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof(clsparseIdx_t), NULL, &status);
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.colIndices" );

        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            (csrMtx.num_rows + 1) * sizeof(clsparseIdx_t), NULL, &status);
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.rowOffsets" );

        if(typeid(T) == typeid(float))
            fileError = clsparseSCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control, explicit_zeroes );
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control, explicit_zeroes );
        else
            fileError = clsparseInvalidType;

        if( fileError != clsparseSuccess )
            throw clsparse::io_exception( "Could not read matrix market data from disk: " + sparseFile );

        clsparseCsrMetaCreate( &csrMtx, control );

        // Initialize the dense X & Y vectors that we multiply against the sparse matrix
        clsparseInitVector( &x );
        x.num_values = csrMtx.num_cols;
        x.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     x.num_values * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer x.values" );

        clsparseInitVector( &y );
        y.num_values = csrMtx.num_rows;
        y.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                     y.num_values * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer y.values" );

        // Initialize the scalar alpha & beta parameters
        clsparseInitScalar( &a );
        a.value = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     1 * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer a.value" );

        clsparseInitScalar( &b );
        b.value = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     1 * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer b.value" );
    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        T scalarOne = 1.0;
        CLSPARSE_V( ::clEnqueueFillBuffer( queue, x.values, &scalarOne, sizeof( T ), 0,
            sizeof( T ) * x.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        T scalarZero = 0.0;
        CLSPARSE_V( ::clEnqueueFillBuffer( queue, y.values, &scalarZero, sizeof( T ), 0,
            sizeof( T ) * y.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );

        CLSPARSE_V( ::clEnqueueFillBuffer( queue, a.value, &alpha, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer alpha.value" );

        CLSPARSE_V( ::clEnqueueFillBuffer( queue, b.value, &beta, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer beta.value" );
    }

    void reset_gpu_write_buffer( )
    {
        T scalar = 0;
        CLSPARSE_V( ::clEnqueueFillBuffer( queue, y.values, &scalar, sizeof( T ), 0,
                             sizeof( T ) * y.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );
    }

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {
        if( gpuTimer && cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
          clsparseIdx_t sparseBytes = sizeof(clsparseIdx_t)*(csrMtx.num_nonzeros + csrMtx.num_rows) + sizeof(T) * (csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows);
          clsparseIdx_t sparseFlops = 2 * csrMtx.num_nonzeros;
          cpuTimer->pruneOutliers( 3.0 );
          cpuTimer->Print( sparseBytes, "GiB/s" );
          cpuTimer->Print( sparseFlops, "GFLOPs" );
          cpuTimer->Reset( );

          gpuTimer->pruneOutliers( 3.0 );
          gpuTimer->Print( sparseBytes, "GiB/s" );
          gpuTimer->Print( sparseFlops, "GFLOPs" );
          gpuTimer->Reset( );
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        clsparseCsrMetaDelete( &csrMtx );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );

        CLSPARSE_V( ::clReleaseMemObject( x.values ), "clReleaseMemObject x.values" );
        CLSPARSE_V( ::clReleaseMemObject( y.values ), "clReleaseMemObject y.values" );
        CLSPARSE_V( ::clReleaseMemObject( a.value ), "clReleaseMemObject alpha.value" );
        CLSPARSE_V( ::clReleaseMemObject( b.value ), "clReleaseMemObject beta.value" );
    }

private:
    void xSpMdV_Function( bool flush );

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
    cldenseVector x;
    cldenseVector y;
    clsparseScalar a;
    clsparseScalar b;

    // host values
    T alpha;
    T beta;
    cl_bool explicit_zeroes;

    //  OpenCL state
    cl_command_queue_properties cqProp;

}; // class xSpMdV

template<> void
xSpMdV<float>::xSpMdV_Function( bool flush )
{
    clsparseStatus status = clsparseScsrmv( &a, &csrMtx, &x, &b, &y, control );

    if( flush )
        clFinish( queue );
}

template<> void
xSpMdV<double>::xSpMdV_Function( bool flush )
{
    clsparseStatus status = clsparseDcsrmv( &a, &csrMtx, &x, &b, &y, control );

    if( flush )
        clFinish( queue );
}

#endif // ifndef CLSPARSE_BENCHMARK_SPMV_HXX__
