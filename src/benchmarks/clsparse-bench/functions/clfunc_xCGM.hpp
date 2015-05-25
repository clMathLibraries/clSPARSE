/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/
#pragma once
#ifndef CLSPARSE_BENCHMARK_CGM_HXX__
#define CLSPARSE_BENCHMARK_CGM_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"

//CG solver benchmark where calculations of scalar values are performed by mapping them on host


template <typename T>
class xCGM : public clsparseFunc
{
public:
    xCGM( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType ):
        clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr )
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

            gpuTimerID = gpuTimer->getUniqueID( "GPU xCGM", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xCGM", 0 );
        }


        clsparseEnableAsync( control, false );

        solverControl = clsparseCreateSolverControl(100, VOID, 1e-6, 1e-8);
    }

    ~xCGM( )
    {
        if( clsparseReleaseSolverControl( solverControl) != clsparseSuccess )
        {
            std::cout << "Problem with releasing solver control object" << std::endl;
        }
    }

    void call_func()
    {
        if( gpuTimer && cpuTimer )
        {
          gpuTimer->Start( gpuTimerID );
          cpuTimer->Start( cpuTimerID );

          xCGM_Function(false);


          cpuTimer->Stop( cpuTimerID );
          gpuTimer->Stop( gpuTimerID );
        }
        else
        {
            xCGM_Function(false);
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
        //TODO:: what is BW per iteration? don't know yet
        return 0;
    }

    std::string bandwidth_formula( )
    {
        return "GiB/s";
    }

    void setup_buffer( double pAlpha, double pBeta, const std::string& path )
    {
        sparseFile = path;


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
        x.n = cooMtx.m;
        x.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                     x.n * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer x.values" );

        clsparseInitVector( &y );
        y.n = cooMtx.n;
        y.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                     y.n * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer y.values" );


    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        // We will solve A*x = y,
        // where initial guess of x will be vector of zeros 0,
        // and y will be vector of ones;

        T xValue = 10.0;
        T yValue = 1.0;

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, x.values, &xValue, sizeof( T ), 0,
            sizeof( T ) * x.n, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, y.values, &yValue, sizeof( T ), 0,
            sizeof( T ) * y.n, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );


    }

    void reset_gpu_write_buffer( )
    {
        // we will solve A*x = y, where initial guess of x will be 0
        T scalar = 0;
        OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, x.values, &scalar, sizeof( T ), 0,
                             sizeof( T ) * x.n, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        // reset solverControl for next call
        clsparseSetSolverParams(solverControl, 100, 1e-6, 1e-8, VOID);
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

          //gpuTimer->pruneOutliers( 3.0 );
          //puTimer->Print( sparseBytes, "GiB/s" );
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


    }

private:
    void xCGM_Function( bool flush );

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


    //  OpenCL state
    cl_command_queue_properties cqProp;

    // solver control object;
    clSParseSolverControl solverControl;
}; //class xCGM

template<> void
xCGM<float>::xCGM_Function( bool flush )
{
    // solve x from y = Ax
    clsparseStatus status = clsparseScsrcg(&x, &csrMtx, &y, solverControl, control);

    if( flush )
        clFinish( queue );
}

template<> void
xCGM<double>::xCGM_Function( bool flush )
{
//    clsparseStatus status = clsparseDcsrmv( &a, &csrMtx, &x, &b, &y, control );

    if( flush )
        clFinish( queue );
}


#endif
