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
#ifndef CLSPARSE_BENCHMARK_COO2CSR_HXX__
#define CLSPARSE_BENCHMARK_COO2CSR_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xCoo2Csr: public clsparseFunc
{
public:
    xCoo2Csr( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_bool explicit_zeroes, cl_device_type devType ): clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr )
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

            gpuTimerID = gpuTimer->getUniqueID( "GPU xCoo2Csr", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xCoo2Csr", 0 );
        }


        clsparseEnableAsync( control, false );
        clsparseEnableExplicitZeroes( control, explicit_zeroes);
    }

    ~xCoo2Csr( )
    {
    }

    void call_func( )
    {
      if( gpuTimer && cpuTimer )
      {
        gpuTimer->Start( gpuTimerID );
        cpuTimer->Start( cpuTimerID );

        xCoo2Csr_Function( false );

        cpuTimer->Stop( cpuTimerID );
        gpuTimer->Stop( gpuTimerID );
      }
      else
      {
        xCoo2Csr_Function( false );
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
        // Number of Elements converted in unit time
        return ( n_vals / time_in_ns( ) );
    }

    std::string bandwidth_formula( )
    {
        return "Gi-Elements/s";
    }


    void setup_buffer( double pAlpha, double pBeta, const std::string& path )
    {
        sparseFile = path;

        cl_int status;

        clsparseStatus fileError = clsparseHeaderfromFile( &n_vals, &n_rows, &n_cols, path.c_str( ) );
        if( fileError != clsparseSuccess )
            throw clsparse::io_exception( "Could not read matrix market header from disk: " + path );

        clsparseInitCooMatrix( &cooMatx );
        cooMatx.num_nonzeros = n_vals;
        cooMatx.num_rows = n_rows;
        cooMatx.num_cols = n_cols;

        cooMatx.values     = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                               cooMatx.num_nonzeros * sizeof(T), NULL, &status );
        cooMatx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                               cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );
        cooMatx.rowIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                               cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );

        if (typeid(T) == typeid(float))
           fileError = clsparseSCooMatrixfromFile(&cooMatx, path.c_str(), control);
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCooMatrixfromFile(&cooMatx, path.c_str(), control);
        else
            fileError = clsparseInvalidType;

        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk: " + path );

        //clsparseCsrMatrix csrMatx;
        clsparseInitCsrMatrix( &csrMtx );

        //JPA:: Shouldn't be CL_MEM_WRITE_ONLY since coo ---> csr???
        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( T ), NULL, &status );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                               cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );
        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_WRITE,
                                              ( cooMatx.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );

    }

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {

    }

    void reset_gpu_write_buffer( )
    {

		int scalar_i = 0;
		T scalar_f = 0;
		CLSPARSE_V( ::clEnqueueFillBuffer( queue, csrMtx.rowOffsets, &scalar_i, sizeof( int ), 0,
                              sizeof( int ) * (csrMtx.num_rows + 1), 0, NULL, NULL ), "::clEnqueueFillBuffer row" );
		CLSPARSE_V( ::clEnqueueFillBuffer( queue, csrMtx.colIndices, &scalar_i, sizeof( int ), 0,
                              sizeof( int ) * csrMtx.num_nonzeros, 0, NULL, NULL ), "::clEnqueueFillBuffer col" );
		CLSPARSE_V( ::clEnqueueFillBuffer( queue, csrMtx.values, &scalar_f, sizeof( T ), 0,
                              sizeof( T ) * csrMtx.num_nonzeros, 0, NULL, NULL ), "::clEnqueueFillBuffer values" );
    }

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {
        if( gpuTimer && cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
          size_t sparseElements = n_vals;
          cpuTimer->pruneOutliers( 3.0 );
          cpuTimer->Print( sparseElements, "GiElements/s" );
          cpuTimer->Reset( );

          gpuTimer->pruneOutliers( 3.0 );
          gpuTimer->Print( sparseElements, "GiElements/s" );
          gpuTimer->Reset( );
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );

        CLSPARSE_V( ::clReleaseMemObject( cooMatx.values ), "clReleaseMemObject cooMtx.values" );
        CLSPARSE_V( ::clReleaseMemObject( cooMatx.colIndices ), "clReleaseMemObject cooMtx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( cooMatx.rowIndices ), "clReleaseMemObject cooMtx.rowOffsets" );
    }

private:
    void xCoo2Csr_Function( bool flush );

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
    clsparseCooMatrix cooMatx;

    //matrix dimension
    int n_rows;
    int n_cols;
    int n_vals;

    //  OpenCL state
    cl_command_queue_properties cqProp;

}; // class xCoo2Csr

template<> void
xCoo2Csr<float>::xCoo2Csr_Function( bool flush )
{
    clsparseStatus status = clsparseScoo2csr(&cooMatx, &csrMtx, control);
    if( flush )
        clFinish( queue );
}

template<> void
xCoo2Csr<double>::xCoo2Csr_Function( bool flush )
{
    clsparseStatus status = clsparseDcoo2csr(&cooMatx, &csrMtx, control);

    if( flush )
        clFinish( queue );
}

#endif // ifndef CLSPARSE_BENCHMARK_COO2CSR_HXX__
