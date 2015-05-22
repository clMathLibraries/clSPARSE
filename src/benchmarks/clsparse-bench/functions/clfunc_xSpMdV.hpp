/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/
#pragma once
#ifndef CLSPARSE_BENCHMARK_SPMV_HXX__
#define CLSPARSE_BENCHMARK_SPMV_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xSpMdV: public clsparseFunc
{
public:
    xSpMdV( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType ): clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr )
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
        return ( sizeof( cl_int )*( csrMtx.nnz + csrMtx.m ) + sizeof( T ) * ( csrMtx.nnz + csrMtx.n + csrMtx.m ) ) / time_in_ns( );
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

        // Create container for sparse data to pass in/out of clsparse API's
        clsparseInitCooMatrix( &cooMtx );

        // Read sparse data from file and construct a COO matrix from it
        clsparseStatus fileError = clsparseCooHeaderfromFile( &cooMtx, sparseFile.c_str( ) );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market header from disk" );

        cl_int status;
        cooMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                          cooMtx.nnz * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer cooMtx.values" );

        cooMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                              cooMtx.nnz * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer cooMtx.colIndices" );

        cooMtx.rowIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                              cooMtx.nnz * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer cooMtx.rowIndices" );

        fileError = clsparseCooMatrixfromFile( &cooMtx, sparseFile.c_str( ), control );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

        // Now initialise a CSR matrix from the COO matrix
        clsparseInitCsrMatrix( &csrMtx );

        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                           cooMtx.nnz * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.values" );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                               cooMtx.nnz * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.colIndices" );

        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                               ( cooMtx.m + 1 ) * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.rowOffsets" );

        clsparseScoo2csr( &csrMtx, &cooMtx, control );

        clsparseCsrMetaSize( &csrMtx, control );

        csrMtx.rowBlocks = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                              csrMtx.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
        clsparseCsrComputeMeta( &csrMtx, control );

        // Initialize the dense X & Y vectors that we multiply against the sparse matrix
        clsparseInitVector( &x );
        x.n = cooMtx.n;
        x.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     x.n * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer x.values" );

        clsparseInitVector( &y );
        y.n = cooMtx.m;
        y.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     y.n * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer y.values" );

        // Initialize the scalar alpha & beta parameters
        clsparseInitScalar( &a );
        a.value = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     1 * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer x.values" );

        clsparseInitScalar( &b );
        b.value = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                     1 * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer y.values" );
    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        T scalarOne = 1.0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, x.values, &scalarOne, sizeof( T ), 0,
            sizeof( T ) * x.n, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        T scalarZero = 0.0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, y.values, &scalarZero, sizeof( T ), 0,
            sizeof( T ) * y.n, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, a.value, &alpha, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer alpha.value" );

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, b.value, &beta, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer beta.value" );
    }

    void reset_gpu_write_buffer( )
    {
        T scalar = 0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, y.values, &scalar, sizeof( T ), 0,
                             sizeof( T ) * y.n, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );
    }

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {	
        if( gpuTimer && cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
          size_t sparseBytes = sizeof( cl_int )*( csrMtx.nnz + csrMtx.m ) + sizeof( T ) * ( csrMtx.nnz + csrMtx.n + csrMtx.m );
          cpuTimer->pruneOutliers( 3.0 );
          cpuTimer->Print( sparseBytes, "GiB/s" );
          cpuTimer->Reset( );

          gpuTimer->pruneOutliers( 3.0 );
          gpuTimer->Print( sparseBytes, "GiB/s" );
          gpuTimer->Reset( );
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        OPENCL_V_THROW( ::clReleaseMemObject( cooMtx.values ), "clReleaseMemObject cooMtx.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( cooMtx.colIndices ), "clReleaseMemObject cooMtx.colIndices" );
        OPENCL_V_THROW( ::clReleaseMemObject( cooMtx.rowIndices ), "clReleaseMemObject cooMtx.rowIndices" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.rowBlocks ), "clReleaseMemObject csrMtx.rowBlocks" );

        OPENCL_V_THROW( ::clReleaseMemObject( x.values ), "clReleaseMemObject x.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( y.values ), "clReleaseMemObject y.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( a.value ), "clReleaseMemObject alpha.value" );
        OPENCL_V_THROW( ::clReleaseMemObject( b.value ), "clReleaseMemObject beta.value" );
    }

private:
    void xSpMdV_Function( bool flush );

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCooMatrix cooMtx;
    clsparseCsrMatrix csrMtx;
    clsparseVector x;
    clsparseVector y;
    clsparseScalar a;
    clsparseScalar b;

    // host values
    T alpha;
    T beta;

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
