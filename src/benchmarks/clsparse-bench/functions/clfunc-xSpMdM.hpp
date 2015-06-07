/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/
#pragma once
#ifndef CLSPARSE_BENCHMARK_SpM_dM_HXX
#define CLSPARSE_BENCHMARK_SpM_dM_HXX

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xSpMdM: public clsparseFunc
{
public:
    xSpMdM( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType, size_t columns ): clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr ), num_columns( columns )
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

            gpuTimerID = gpuTimer->getUniqueID( "GPU xSpMdM", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xSpMdM", 0 );
        }


        clsparseEnableAsync( control, false );
    }

    ~xSpMdM( )
    {
    }

    void call_func( )
    {
      if( gpuTimer && cpuTimer )
      {
        gpuTimer->Start( gpuTimerID );
        cpuTimer->Start( cpuTimerID );

        xSpMdM_Function( false );

        cpuTimer->Stop( cpuTimerID );
        gpuTimer->Stop( gpuTimerID );
      }
      else
      {
        xSpMdM_Function( false );
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

        // Read sparse data from file and construct a CSR matrix from it
        int nnz, row, col;
        clsparseStatus fileError = clsparseHeaderfromFile( &nnz, &row, &col, sparseFile.c_str( ) );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market header from disk" );

        // Now initialise a CSR matrix from the CSR matrix
        clsparseInitCsrMatrix( &csrMtx );
        csrMtx.nnz = nnz;
        csrMtx.m = row;
        csrMtx.n = col;
        clsparseCsrMetaSize( &csrMtx, control );

        cl_int status;
        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY, csrMtx.nnz * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.values" );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY, csrMtx.nnz * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.colIndices" );

        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY, ( csrMtx.m + 1 ) * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.rowOffsets" );

        csrMtx.rowBlocks = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY, csrMtx.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.rowBlocks" );

        fileError = clsparseSCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

        // Initialize the dense B & C matrices that we multiply against the sparse matrix
        // We are shaping B, such that no matter what shape A is, C will result in a square matrix
        cldenseInitMatrix( &denseB );
        denseB.major = rowMajor;
        denseB.num_rows = csrMtx.n;
        denseB.num_cols = num_columns;
        denseB.lead_dim = denseB.num_cols;
        denseB.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                          denseB.num_rows * denseB.lead_dim * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer denseB.values" );

        //  C is square because the B columns is equal to the A rows
        cldenseInitMatrix( &denseC );
        denseC.major = rowMajor;
        denseC.num_rows = csrMtx.m;
        denseC.num_cols = num_columns;
        denseC.lead_dim = denseC.num_cols;
        denseC.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                          denseC.num_rows * denseC.lead_dim * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer denseC.values" );

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
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, denseB.values, &scalarOne, sizeof( T ), 0,
            denseB.num_rows * denseB.num_cols * sizeof( T ), 0, NULL, NULL ), "::clEnqueueFillBuffer denseB.values" );

        T scalarZero = 0.0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, denseC.values, &scalarZero, sizeof( T ), 0,
            denseC.num_rows * denseC.num_cols * sizeof( T ), 0, NULL, NULL ), "::clEnqueueFillBuffer denseC.values" );

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, a.value, &alpha, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer alpha.value" );

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, b.value, &beta, sizeof( T ), 0,
            sizeof( T ) * 1, 0, NULL, NULL ), "::clEnqueueFillBuffer beta.value" );
    }

    void reset_gpu_write_buffer( )
    {
        T scalar = 0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, denseC.values, &scalar, sizeof( T ), 0,
            denseC.num_rows * denseC.num_cols * sizeof( T ), 0, NULL, NULL ), "::clEnqueueFillBuffer denseC.values" );
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
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMtx.rowBlocks ), "clReleaseMemObject csrMtx.rowBlocks" );

        OPENCL_V_THROW( ::clReleaseMemObject( denseB.values ), "clReleaseMemObject x.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( denseC.values ), "clReleaseMemObject y.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( a.value ), "clReleaseMemObject alpha.value" );
        OPENCL_V_THROW( ::clReleaseMemObject( b.value ), "clReleaseMemObject beta.value" );
    }

private:
    void xSpMdM_Function( bool flush );

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
    cldenseMatrix denseB;
    cldenseMatrix denseC;
    clsparseScalar a;
    clsparseScalar b;

    // host values
    T alpha;
    T beta;
    size_t num_columns;

    //  OpenCL state
    cl_command_queue_properties cqProp;

}; // class xSpMdM

template<> void
xSpMdM<float>::xSpMdM_Function( bool flush )
{
    clsparseStatus status = clsparseScsrmm( &a, &csrMtx, &denseB, &b, &denseC, control );

    if( flush )
        clFinish( queue );
}

template<> void
xSpMdM<double>::xSpMdM_Function( bool flush )
{
    //clsparseStatus status = clsparseDcsrmm( &a, &csrMtx, &denseB, &b, &denseC, control );

    //if( flush )
    //    clFinish( queue );
}

#endif // ifndef CLSPARSE_BENCHMARK_SpM_dM_HXX
