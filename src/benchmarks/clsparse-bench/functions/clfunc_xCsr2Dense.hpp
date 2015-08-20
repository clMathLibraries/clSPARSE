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

#ifndef CLSPARSE_BENCHMARK_xCsr2Dense_HXX
#define CLSPARSE_BENCHMARK_xCsr2Dense_HXX

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xCsr2Dense : public clsparseFunc
{
public:
    xCsr2Dense(PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type dev_type) : clsparseFunc(dev_type, CL_QUEUE_PROFILING_ENABLE)
    {
        gpuTimer = nullptr;
        cpuTimer = nullptr;

        // Create and initialize timer class, if the external timer shared library loaded
        if (sparseGetTimer)
        {
            gpuTimer = sparseGetTimer( CLSPARSE_GPU );
            gpuTimer->Reserve(1, profileCount);
            gpuTimer->setNormalize(true);

            cpuTimer = sparseGetTimer(CLSPARSE_CPU);
            cpuTimer->Reserve(1, profileCount);
            cpuTimer->setNormalize(true);

            gpuTimerID = gpuTimer->getUniqueID("GPU xCsr2Dense", 0);
            cpuTimerID = cpuTimer->getUniqueID("CPU xCsr2Dense", 0);
        }

        clsparseEnableAsync(control, false);
    }// End of constructor

    ~xCsr2Dense()
    {

    }

    void call_func()
    {
        if (gpuTimer && cpuTimer)
        {
            gpuTimer->Start(gpuTimerID);
            cpuTimer->Start(cpuTimerID);

            xCsr2Dense_Function(false);

            cpuTimer->Stop(cpuTimerID);
            gpuTimer->Stop(gpuTimerID);
        }
        else
        {
            xCsr2Dense_Function(false);
        }
    }// end of call_func()

    double gflops()
    {
        return 0.0;
    }// end

    std::string gflops_formula()
    {
        return "N/A";
    }//end

    double bandwidth()
    {
#if 0
        //Check VK
		//Host to GPU: CSR-> [rowOffsets(num_rows + 1) + Column Indices] * sizeof(int) + sizeof(T) * (num_nonzero)
		//GPU to Host: Dense - > [sizeof(T) * denseMtx.num_rows * denseMTx.num_cols]
		size_t sparseBytes = sizeof(cl_int) * (csrMtx.num_nonzeros + csrMtx.num_rows + 1) + sizeof(T) * (csrMtx.num_nonzeros) + sizeof(T) * (denseMtx.num_rows * denseMtx.num_cols);
        return (sparseBytes / time_in_ns());
#endif
		// Number of Elements converted in unit time
		return (csrMtx.num_nonzeros / time_in_ns());
    }// end

    std::string bandwidth_formula()
    {
        //return "GiB/s";
		return "GiElements/s";
    }// end

    void setup_buffer(double pAlpha, double pBeta, const std::string& path)
    {
        sparseFile = path;

        // Read sparse data from file and construct a CSR matrix from it
        int nnz;
        int row;
        int col;
        clsparseStatus fileError = clsparseHeaderfromFile(&nnz, &row, &col, sparseFile.c_str());
        if (clsparseSuccess != fileError)
            throw std::runtime_error("Could not read matrix market header from disk");

        // Now initialize a CSR matrix from the CSR matrix
        // VK we have to handle other cases if input mtx file is not in CSR format
        clsparseInitCsrMatrix(&csrMtx);
        csrMtx.num_nonzeros = nnz;
        csrMtx.num_rows     = row;
        csrMtx.num_cols     = col;
        clsparseCsrMetaSize( &csrMtx, control );

        cl_int status;
        csrMtx.values = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, csrMtx.num_nonzeros * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.values");

        csrMtx.colIndices = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, csrMtx.num_nonzeros * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.colIndices");

        csrMtx.rowOffsets = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, (csrMtx.num_rows + 1) * sizeof(cl_int), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.rowOffsets");

        csrMtx.rowBlocks = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, csrMtx.rowBlockSize * sizeof(cl_ulong), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer csrMtx.rowBlocks");

		if (typeid(T) == typeid(float))
			fileError = clsparseSCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control);
		else if (typeid(T) == typeid(double))
			fileError = clsparseDCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control);
		else
			fileError = clsparseInvalidType;

        if (fileError != clsparseSuccess)
            throw std::runtime_error("Could not read matrix market data from disk");

        // Initialize the output dense matrix
        cldenseInitMatrix(&denseMtx);
        denseMtx.major    = rowMajor;
        denseMtx.num_rows = row;
        denseMtx.num_cols = col;
		denseMtx.lead_dim = col;  // To Check!! VK;
        denseMtx.values = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
                                            denseMtx.num_rows * denseMtx.num_cols * sizeof(T), NULL, &status);
        CLSPARSE_V(status, "::clCreateBuffer denseMtx.values");
    }// end

    void initialize_cpu_buffer()
    {
    }//end

    void initialize_gpu_buffer()
    {
        T scalarZero = 0.0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, denseMtx.values, &scalarZero, sizeof(T), 0,
            denseMtx.num_rows * denseMtx.num_cols * sizeof(T), 0, NULL, NULL), "::clEnqueueFillBuffer denseMtx.values");

    }// end

    void reset_gpu_write_buffer()
    {
        T scalar = 0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, denseMtx.values, &scalar, sizeof(T), 0,
            denseMtx.num_rows * denseMtx.num_cols * sizeof(T), 0, NULL, NULL), "::clEnqueueFillBuffer denseMtx.values");
    }// end
    void read_gpu_buffer()
    {
    }//end

    void cleanup(void)
    {
        if (gpuTimer && cpuTimer)
        {
            std::cout << "clSPARSE matrix: " << sparseFile << std::endl;
#if 0
            // Need to verify this calculation VK
            //size_t sparseBytes = sizeof(cl_int) * (csrMtx.nnz + csrMtx.m) + sizeof(T) * (csrMtx.nnz + csrMtx.n + csrMtx.m);
			//Host to GPU: CSR-> [rowOffsets(num_rows + 1) + Column Indices] * sizeof(int) + sizeof(T) * (num_nonzero)
			//GPU to Host: Dense - > [sizeof(T) * denseMtx.num_rows * denseMTx.num_cols]
            size_t sparseBytes = sizeof(cl_int) * (csrMtx.num_nonzeros + csrMtx.num_rows + 1) + sizeof(T) * (csrMtx.num_nonzeros) + sizeof(T) * (denseMtx.num_rows * denseMtx.num_cols);
            cpuTimer->pruneOutliers(3.0);
            cpuTimer->Print( sparseBytes, "GiB/s" );
            cpuTimer->Reset();

            gpuTimer->pruneOutliers( 3.0 );
            gpuTimer->Print( sparseBytes, "GiB/s" );
            gpuTimer->Reset();
#endif
			// Calculate Number of Elements transformed per unit time
			size_t sparseElements = csrMtx.num_nonzeros;
			cpuTimer->pruneOutliers(3.0);
			cpuTimer->Print(sparseElements, "GiElements/s");
			cpuTimer->Reset();

			gpuTimer->pruneOutliers(3.0);
			gpuTimer->Print(sparseElements, "GiElements/s");
			gpuTimer->Reset();
        }

        //this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
        //need to do this before we eventually hit the destructor
        CLSPARSE_V(::clReleaseMemObject(csrMtx.values), "clReleaseMemObject csrMtx.values");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.colIndices), "clReleaseMemObject csrMtx.colIndices");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.rowOffsets), "clReleaseMemObject csrMtx.rowOffsets");
        CLSPARSE_V(::clReleaseMemObject(csrMtx.rowBlocks), "clReleaseMemObject csrMtx.rowBlocks");

        CLSPARSE_V(::clReleaseMemObject(denseMtx.values), "clReleaseMemObject denseMtx.values");
    }

private:
    void xCsr2Dense_Function(bool flush);

    // Timers
    clsparseTimer* gpuTimer;
    clsparseTimer* cpuTimer;
    size_t gpuTimerID;
    size_t cpuTimerID;

    std::string sparseFile;

    //device values
    clsparseCsrMatrix csrMtx;
    cldenseMatrix     denseMtx;

    //host values


    //OpenCL state
    cl_command_queue_properties cqProp;

}; // class xCsr2Dense

template<> void
xCsr2Dense<float>::xCsr2Dense_Function(bool flush)
{
	clsparseStatus status = clsparseScsr2dense(&csrMtx, &denseMtx, control);
	if (flush)
		clFinish(queue);
}// end


template<> void
xCsr2Dense<double>::xCsr2Dense_Function(bool flush)
{
	clsparseStatus status = clsparseDcsr2dense(&csrMtx, &denseMtx, control);
	if (flush)
		clFinish(queue);
}



#endif // ifndef CLSPARSE_BENCHMARK_xCsr2Dense_HXX
