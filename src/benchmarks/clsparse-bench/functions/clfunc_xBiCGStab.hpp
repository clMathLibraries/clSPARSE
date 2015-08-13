/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
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
#ifndef CLSPARSE_BENCHMARK_BICGSTAB_HXX__
#define CLSPARSE_BENCHMARK_BICGSTAB_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"
#include <vector>



template <typename T>
class xBiCGStab : public clsparseFunc
{
public:
    xBiCGStab( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType ):
        clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ),/* gpuTimer( nullptr ),*/ cpuTimer( nullptr )
    {
        //	Create and initialize our timer class, if the external timer shared library loaded
        if( sparseGetTimer )
        {
//            gpuTimer = sparseGetTimer( CLSPARSE_GPU );
//            gpuTimer->Reserve( 1, profileCount );
//            gpuTimer->setNormalize( true );

            cpuTimer = sparseGetTimer( CLSPARSE_CPU );
            cpuTimer->Reserve( 1, profileCount );
            cpuTimer->setNormalize( true );

//            gpuTimerID = gpuTimer->getUniqueID( "GPU xCGM", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xBiCGStab", 0 );
        }


        clsparseEnableAsync( control, false );

        solverControl = clsparseCreateSolverControl(DIAGONAL, 1000, 1e-6, 0);
        clsparseSolverPrintMode(solverControl, VERBOSE);
    }

    ~xBiCGStab( )
    {
        if( clsparseReleaseSolverControl( solverControl) != clsparseSuccess )
        {
            std::cout << "Problem with releasing solver control object" << std::endl;
        }
    }

    void call_func()
    {
        if( /*gpuTimer && */cpuTimer )
        {
//          gpuTimer->Start( gpuTimerID );
          cpuTimer->Start( cpuTimerID );

          xBiCGStab_Function(false);


          cpuTimer->Stop( cpuTimerID );
//          gpuTimer->Stop( gpuTimerID );
        }
        else
        {
            xBiCGStab_Function(false);
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

        // Read sparse data from file and construct a COO matrix from it
        int nnz, row, col;
        clsparseStatus fileError = clsparseHeaderfromFile( &nnz, &row, &col, sparseFile.c_str( ) );
        if( fileError != clsparseSuccess )
            throw clsparse::io_exception( "Could not read matrix market header from disk" );

        // Now initialise a CSR matrix from the COO matrix
        clsparseInitCsrMatrix( &csrMtx );
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows = row;
        csrMtx.num_cols = col;
        clsparseCsrMetaSize( &csrMtx, control );

        cl_int status;
        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.values" );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof( cl_int ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.colIndices" );

        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            ( csrMtx.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.rowOffsets" );

        csrMtx.rowBlocks = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer csrMtx.rowBlocks" );

        if(typeid(T) == typeid(float))
            fileError = clsparseSCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else
            fileError = clsparseInvalidType;

        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

        // Initialize the dense X & Y vectors that we multiply against the sparse matrix
        clsparseInitVector( &x );
        x.num_values = csrMtx.num_rows;
        x.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                     x.num_values * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer x.values" );

        clsparseInitVector( &y );
        y.num_values = csrMtx.num_cols;
        y.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                     y.num_values * sizeof( T ), NULL, &status );
        CLSPARSE_V( status, "::clCreateBuffer y.values" );


    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
        // We will solve A*x = y,
        // where initial guess of x will be vector of zeros 0,
        // and y will be vector of ones;

        T xValue = 0.0;
        T yValue = 1.0;

        CLSPARSE_V( ::clEnqueueFillBuffer( queue, x.values, &xValue, sizeof( T ), 0,
            sizeof( T ) * x.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        CLSPARSE_V( ::clEnqueueFillBuffer( queue, y.values, &yValue, sizeof( T ), 0,
            sizeof( T ) * y.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer y.values" );


    }

    void reset_gpu_write_buffer( )
    {
        // we will solve A*x = y, where initial guess of x will be 0
        T scalar = 0;
        CLSPARSE_V( ::clEnqueueFillBuffer( queue, x.values, &scalar, sizeof( T ), 0,
                             sizeof( T ) * x.num_values, 0, NULL, NULL ), "::clEnqueueFillBuffer x.values" );

        // reset solverControl for next call
        clsparseSetSolverParams(solverControl, NOPRECOND, 100, 1e-2, 1e-8);
    }

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {
        if(/* gpuTimer && */cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
          size_t sparseBytes = sizeof( cl_int )*( csrMtx.num_nonzeros + csrMtx.num_rows ) + sizeof( T ) * ( csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows );
          cpuTimer->pruneOutliers( 3.0 );
          cpuTimer->Print( sparseBytes, "GiB/s" );
          cpuTimer->Reset( );

          //gpuTimer->pruneOutliers( 3.0 );
          //puTimer->Print( sparseBytes, "GiB/s" );
//          gpuTimer->Reset( );
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.rowBlocks ), "clReleaseMemObject csrMtx.rowBlocks" );

        CLSPARSE_V( ::clReleaseMemObject( x.values ), "clReleaseMemObject x.values" );
        CLSPARSE_V( ::clReleaseMemObject( y.values ), "clReleaseMemObject y.values" );


    }

private:
    void xBiCGStab_Function( bool flush );

    //  Timers we want to keep
   // clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
    cldenseVector x;
    cldenseVector y;


    //  OpenCL state
    cl_command_queue_properties cqProp;

    // solver control object;
    clSParseSolverControl solverControl;
}; //class xCGM

template<> void
xBiCGStab<float>::xBiCGStab_Function( bool flush )
{
    // solve x from y = Ax
    try {
    clsparseStatus status = clsparseScsrbicgStab(&x, &csrMtx, &y, solverControl, control);

//    std::vector<float> h_y(x.num_values);
//    clEnqueueReadBuffer(queue, x.values, CL_TRUE, 0, x.num_values * sizeof(float), h_y.data(), 0, NULL, NULL );
//    for (int i = 0; i < h_y.size(); i++)
//    {
//        std::cout << "hy [" << i << "] = " << h_y[i] << std::endl;
//    }
    }
    catch (std::out_of_range e)
    {
        std::cout << "e: " << std::endl;
    }
    catch (...)
    {
        std::cout << "xxx" << std::endl;
    }

    if( flush )
        clFinish( queue );
}

template<> void
xBiCGStab<double>::xBiCGStab_Function( bool flush )
{
    clsparseStatus status = clsparseDcsrbicgStab(&x, &csrMtx, &y, solverControl, control);

    if( flush )
        clFinish( queue );
}


#endif
