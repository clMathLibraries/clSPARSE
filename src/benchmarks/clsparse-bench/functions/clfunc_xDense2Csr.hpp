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
#ifndef CLSPARSE_BENCHMARK_DENSE2CSR_HXX__
#define CLSPARSE_BENCHMARK_DENSE2CSR_HXX__

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xDense2Csr: public clsparseFunc
{
public:
    xDense2Csr( PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type devType ): clsparseFunc( devType, CL_QUEUE_PROFILING_ENABLE ), gpuTimer( nullptr ), cpuTimer( nullptr )
    {
		gpuTimer = nullptr;
		cpuTimer = nullptr;

        //      Create and initialize our timer class, if the external timer shared library loaded
        if( sparseGetTimer )
        {
            gpuTimer = sparseGetTimer( CLSPARSE_GPU );
            gpuTimer->Reserve( 1, profileCount );
            gpuTimer->setNormalize( true );

            cpuTimer = sparseGetTimer( CLSPARSE_CPU );
            cpuTimer->Reserve( 1, profileCount );
            cpuTimer->setNormalize( true );

            gpuTimerID = gpuTimer->getUniqueID( "GPU xDense2Csr", 0 );
            cpuTimerID = cpuTimer->getUniqueID( "CPU xDense2Csr", 0 );
        }
        clsparseEnableAsync( control, false );
    }// End of constructor

    ~xDense2Csr( )
    {
    }

    void call_func( )
    {
      if( gpuTimer && cpuTimer )
      {
        gpuTimer->Start( gpuTimerID );
        cpuTimer->Start( cpuTimerID );

        xDense2Csr_Function( false );

        cpuTimer->Stop( cpuTimerID );
        gpuTimer->Stop( gpuTimerID );
      }
      else
      {
        xDense2Csr_Function( false );
      }
    }// end of call_func

    double gflops( )
    {
        return 0.0;
    }// end

    std::string gflops_formula( )
    {
        return "N/A";
    }// end

    double bandwidth( )
    {
#if 0
		//  Assuming that accesses to the vector always hit in the cache after the first access
        //  There are NNZ integers in the cols[ ] array
        //  You access each integer value in row_delimiters[ ] once.
        //  There are NNZ float_types in the vals[ ] array
        //  You read num_cols floats from the vector, afterwards they cache perfectly.
        //  Finally, you write num_rows floats out to DRAM at the end of the kernel.
        return ( sizeof( cl_int )*( csrMtx.num_nonzeros + csrMtx.num_rows ) + sizeof( T ) * ( csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows ) ) / time_in_ns( );
#endif
		// Number of Elements converted in unit time
		return (csrMtx.num_cols * csrMtx.num_rows / time_in_ns());
    }

    std::string bandwidth_formula( )
    {
        //return "GiB/s";
		return "GiElements/s";
    }


    void setup_buffer( double pAlpha, double pBeta, const std::string& path )
    {
        sparseFile = path;

        // Read sparse data from file and construct a COO matrix from it
		int nnz;
		int row;
		int col;
        clsparseStatus fileError = clsparseHeaderfromFile( &nnz, &row, &col, sparseFile.c_str( ) );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market header from disk" );

        // Now initialise a CSR matrix from the COO matrix
        clsparseInitCsrMatrix( &csrMtx );
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows     = row;
        csrMtx.num_cols     = col;

		//clsparseCsrMetaSize( &csrMtx, control );

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

        if(typeid(T) == typeid(float))
            fileError = clsparseSCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else
            fileError = clsparseInvalidType;

        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

		// Initialize the input dense matrix
		cldenseInitMatrix(&A);
		A.major    = rowMajor;
		A.num_rows = row;
		A.num_cols = col;
		A.lead_dim = col;  // To Check!! VK;
		A.values = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
			                      row * col * sizeof(T), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer A.values");

        if(typeid(T) == typeid(float))
		{
			clsparseScsr2dense(&csrMtx, &A, control);
		}
		else if (typeid(T) == typeid(double))
		{
			clsparseDcsr2dense(&csrMtx, &A, control);
		}
		// Check whether we need any barrier here: clFinish(queue)

		// Output CSR Matrix
		clsparseInitCsrMatrix(&csrMatx);
		csrMatx.num_cols     = csrMtx.num_cols;
		csrMatx.num_rows     = csrMtx.num_rows;
		csrMatx.num_nonzeros = csrMtx.num_nonzeros;

        csrMatx.values = ::clCreateBuffer( ctx, CL_MEM_WRITE_ONLY,
                                           csrMtx.num_nonzeros * sizeof( T ), NULL, &status );
		CLSPARSE_V(status, "::clCreateBuffer csrMatx.values");

        csrMatx.colIndices = ::clCreateBuffer( ctx, CL_MEM_WRITE_ONLY,
                                           csrMtx.num_nonzeros * sizeof( cl_int ), NULL, &status );
		CLSPARSE_V(status, "::clCreateBuffer csrMatx.colIndices");

        csrMatx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_WRITE_ONLY,
                                           (csrMtx.num_rows + 1) * sizeof( cl_int ), NULL, &status );
		CLSPARSE_V(status, "::clCreateBuffer csrMatx.rowOffsets");
    }// End of function

    void initialize_cpu_buffer( )
    {
    }

    void initialize_gpu_buffer( )
    {
    }

    void reset_gpu_write_buffer( )
    {
		int scalar_i = 0;
		T scalar_f   = 0;

		CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMatx.rowOffsets, &scalar_i, sizeof(int), 0,
			sizeof(int) * (csrMatx.num_rows + 1), 0, NULL, NULL), "::clEnqueueFillBuffer row");

		CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMatx.colIndices, &scalar_i, sizeof(int), 0,
			sizeof(int) * csrMatx.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer col");

		CLSPARSE_V(::clEnqueueFillBuffer(queue, csrMatx.values, &scalar_f, sizeof(T), 0,
			sizeof(T) * csrMatx.num_nonzeros, 0, NULL, NULL), "::clEnqueueFillBuffer values");
    }// end

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {
        if( gpuTimer && cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
#if 0
          size_t sparseBytes = 0;
          cpuTimer->pruneOutliers( 3.0 );
          cpuTimer->Print( sparseBytes, "GiB/s" );
          cpuTimer->Reset( );

          gpuTimer->pruneOutliers( 3.0 );
          gpuTimer->Print( sparseBytes, "GiB/s" );
          gpuTimer->Reset( );
#endif
		  // Calculate Number of Elements transformed per unit time
		  size_t sparseElements = A.num_cols * A.num_rows;
		  cpuTimer->pruneOutliers(3.0);
		  cpuTimer->Print(sparseElements, "GiElements/s");
		  cpuTimer->Reset();

		  gpuTimer->pruneOutliers(3.0);
		  gpuTimer->Print(sparseElements, "GiElements/s");
		  gpuTimer->Reset();
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.values ), "clReleaseMemObject csrMtx.values" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( csrMtx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );

        CLSPARSE_V( ::clReleaseMemObject( csrMatx.values ), "clReleaseMemObject csrMatx.values" );
        CLSPARSE_V( ::clReleaseMemObject( csrMatx.colIndices ), "clReleaseMemObject csrMatx.colIndices" );
        CLSPARSE_V( ::clReleaseMemObject( csrMatx.rowOffsets ), "clReleaseMemObject csrMatx.rowOffsets" );

		CLSPARSE_V( ::clReleaseMemObject( A.values ), "clReleaseMemObject A.values" );
    }// End of function

private:
    void xDense2Csr_Function( bool flush );

    //  Timers we want to keep
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID, cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
	clsparseCsrMatrix csrMatx;
    cldenseMatrix A;

    //  OpenCL state
    cl_command_queue_properties cqProp;

}; // class xDense2Csr

template<> void
xDense2Csr<float>::xDense2Csr_Function( bool flush )
{
      //call dense2csr
    clsparseSdense2csr(&csrMatx, &A, control);

	if( flush )
        clFinish( queue );
}// end

template<> void
xDense2Csr<double>::xDense2Csr_Function( bool flush )
{
     //call dense2csr
    clsparseDdense2csr(&csrMatx, &A, control);

    if( flush )
        clFinish( queue );
}// end

#endif // ifndef CLSPARSE_BENCHMARK_DENSE2CSR_HXX__
