/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/
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
    }

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
        return ( sizeof( cl_int )*( csrMtx.num_nonzeros + csrMtx.num_rows ) + sizeof( T ) * ( csrMtx.num_nonzeros + csrMtx.num_cols + csrMtx.num_rows ) ) / time_in_ns( );
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
            throw std::runtime_error( "Could not read matrix market header from disk" );

        // Now initialise a CSR matrix from the COO matrix
        clsparseInitCsrMatrix( &csrMtx );
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows = row;
        csrMtx.num_cols = col;

        cl_int status;
        csrMtx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof( T ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.values" );

        csrMtx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            csrMtx.num_nonzeros * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.colIndices" );

        csrMtx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
            ( csrMtx.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );
        OPENCL_V_THROW( status, "::clCreateBuffer csrMtx.rowOffsets" );

        if(typeid(T) == typeid(float))
            fileError = clsparseSCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else if (typeid(T) == typeid(double))
            fileError = clsparseDCsrMatrixfromFile( &csrMtx, sparseFile.c_str( ), control );
        else
            fileError = clsparseInvalidType;

        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

	cldenseInitMatrix(&A);
	A.values = clCreateBuffer(ctx,
                                  CL_MEM_READ_WRITE,
                                  row * col * sizeof(T), NULL, &status);
								  
	A.num_rows = row;
        A.num_cols = col;
      
        if(typeid(T) == typeid(float))
          clsparseScsr2dense(&csrMtx,
                         &A,
                         control);
        if(typeid(T) == typeid(double))
	   clsparseDcsr2dense(&csrMtx,
                         &A,
                         control);				   
          

	clsparseInitCsrMatrix( &csrMatx );

        csrMatx.values = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                           csrMtx.num_nonzeros * sizeof( T ), NULL, &status );
        csrMatx.colIndices = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                           csrMtx.num_nonzeros * sizeof( cl_int ), NULL, &status );
        csrMatx.rowOffsets = ::clCreateBuffer( ctx, CL_MEM_READ_ONLY,
                                           (csrMtx.num_rows + 1) * sizeof( cl_int ), NULL, &status );
					   
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
                OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, csrMatx.rowOffsets, &scalar_i, sizeof( int ), 0,
                              sizeof( int ) * (csrMatx.num_rows + 1), 0, NULL, NULL ), "::clEnqueueFillBuffer row" );
                OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, csrMatx.colIndices, &scalar_i, sizeof( int ), 0,
                              sizeof( int ) * csrMatx.num_nonzeros, 0, NULL, NULL ), "::clEnqueueFillBuffer col" );
                OPENCL_V_THROW( ::clEnqueueFillBuffer( queue, csrMatx.values, &scalar_f, sizeof( T ), 0,
                              sizeof( T ) * csrMatx.num_nonzeros, 0, NULL, NULL ), "::clEnqueueFillBuffer values" );
    }

    void read_gpu_buffer( )
    {
    }

    void cleanup( )
    {
        if( gpuTimer && cpuTimer )
        {
          std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
          size_t sparseBytes = 0;
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

        OPENCL_V_THROW( ::clReleaseMemObject( csrMatx.values ), "clReleaseMemObject csrMtx.values" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMatx.colIndices ), "clReleaseMemObject csrMtx.colIndices" );
        OPENCL_V_THROW( ::clReleaseMemObject( csrMatx.rowOffsets ), "clReleaseMemObject csrMtx.rowOffsets" );
		
		OPENCL_V_THROW( ::clReleaseMemObject( A.values ), "clReleaseMemObject A.values" );
		
    }

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

    //matrix dimension
    int m;
    int n;
    int nnz;

    //  OpenCL state
    cl_command_queue_properties cqProp;

}; // class xDense2Csr

template<> void
xDense2Csr<float>::xDense2Csr_Function( bool flush )
{
      //call dense2csr
    clsparseSdense2csr(&csrMatx, &A,
                       control);
    if( flush )
        clFinish( queue );
}

template<> void
xDense2Csr<double>::xDense2Csr_Function( bool flush )
{
    //clsparseStatus status = clsparseDcoo2csr(&cooMatx, &csrMtx, control);
     //call dense2csr
    clsparseDdense2csr(&csrMatx, &A,
                       control);

    if( flush )
        clFinish( queue );
}

#endif // ifndef CLSPARSE_BENCHMARK_DENSE2CSR_HXX__

