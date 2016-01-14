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

#ifndef CLSPARSE_BENCHMARK_xCsr2Coo_HXX
#define CLSPARSE_BENCHMARK_xCsr2Coo_HXX

#include "clSPARSE.h"
#include "clfunc_common.hpp"

template <typename T>
class xCsr2Coo : public clsparseFunc
{
public:
	xCsr2Coo(PFCLSPARSETIMER sparseGetTimer, size_t profileCount, cl_device_type dev_type, cl_bool keep_explicit_zeroes = true) : clsparseFunc(dev_type, CL_QUEUE_PROFILING_ENABLE)
	{
		gpuTimer = nullptr;
		cpuTimer = nullptr;

		// Create and initialize timer class, if the external timer shared library loaded
		if (sparseGetTimer)
		{
			gpuTimer = sparseGetTimer(CLSPARSE_GPU);
			gpuTimer->Reserve(1, profileCount);
			gpuTimer->setNormalize(true);

			cpuTimer = sparseGetTimer(CLSPARSE_CPU);
			cpuTimer->Reserve(1, profileCount);
			cpuTimer->setNormalize(true);

			gpuTimerID = gpuTimer->getUniqueID("GPU xCsr2Coo", 0);
			cpuTimerID = cpuTimer->getUniqueID("CPU xCsr2Coo", 0);
		}

		clsparseEnableAsync(control, false);
                explicit_zeroes = keep_explicit_zeroes;
	}// End of constructor

	~xCsr2Coo()
	{
	}

	void call_func()
	{
		if (gpuTimer && cpuTimer)
		{
			gpuTimer->Start(gpuTimerID);
			cpuTimer->Start(cpuTimerID);

			xCsr2Coo_Function(false);

			cpuTimer->Stop(cpuTimerID);
			gpuTimer->Stop(gpuTimerID);
		}
		else
		{
			xCsr2Coo_Function(false);
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
		//Host to GPU: CSR-> [row_pointer(num_rows + 1) + Column Indices] * sizeof(int) + sizeof(T) * (num_nonzero)
		//GPU to Host: Coo - > row_indices + Col_indices + Values- > [sizeof(T) * num_nonzero] + sizeof(int)
		size_t sparseBytes = sizeof(cl_int) * (csrMtx.num_nonzeros + csrMtx.num_rows + 1) + sizeof(T) * (csrMtx.num_nonzeros) +
			sizeof(T) * (cooMtx.num_nonzeros) + sizeof(cl_int) * (cooMtx.num_nonzeros * 2);
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
        clsparseIdx_t nnz;
        clsparseIdx_t row;
        clsparseIdx_t col;
		clsparseStatus fileError = clsparseHeaderfromFile(&nnz, &row, &col, sparseFile.c_str());
		if (clsparseSuccess != fileError)
             throw clsparse::io_exception( "Could not read matrix market header from disk: " + sparseFile );

		// Now initialize a CSR matrix from the CSR matrix
		// VK we have to handle other cases if input mtx file is not in CSR format
		clsparseInitCsrMatrix(&csrMtx);
		csrMtx.num_nonzeros = nnz;
		csrMtx.num_rows     = row;
		csrMtx.num_cols     = col;

		cl_int status;
		csrMtx.values = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, csrMtx.num_nonzeros * sizeof(T), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer csrMtx.values");

        csrMtx.col_indices = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, csrMtx.num_nonzeros * sizeof(clsparseIdx_t), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer csrMtx.col_indices");

        csrMtx.row_pointer = ::clCreateBuffer(ctx, CL_MEM_READ_ONLY, (csrMtx.num_rows + 1) * sizeof(clsparseIdx_t), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer csrMtx.row_pointer");

		if (typeid(T) == typeid(float))
			fileError = clsparseSCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control, explicit_zeroes);
		else if (typeid(T) == typeid(double))
			fileError = clsparseDCsrMatrixfromFile(&csrMtx, sparseFile.c_str(), control, explicit_zeroes);
		else
			fileError = clsparseInvalidType;

		if (fileError != clsparseSuccess)
            throw std::runtime_error("Could not read matrix market data from disk: " + sparseFile);

		clsparseCsrMetaCreate(&csrMtx, control);

		// Initialize the output coo matrix
		clsparseInitCooMatrix(&cooMtx);
		cooMtx.num_rows     = csrMtx.num_rows;
		cooMtx.num_cols     = csrMtx.num_cols;
		cooMtx.num_nonzeros = csrMtx.num_nonzeros;

		cooMtx.values = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,  //CL_MEM_READ_ONLY
			cooMtx.num_nonzeros * sizeof(T), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer cooMtx.values");

		cooMtx.col_indices = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer cooMtx.col_indices");

		cooMtx.row_indices = ::clCreateBuffer(ctx, CL_MEM_WRITE_ONLY,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), NULL, &status);
		CLSPARSE_V(status, "::clCreateBuffer cooMtx.row_indices");

	}// end

	void initialize_cpu_buffer()
	{
	}//end

	void initialize_gpu_buffer()
	{
		T scalarZero = 0.0;
		CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.values, &scalarZero, sizeof(T), 0,
			cooMtx.num_nonzeros * sizeof(T), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.values");

        clsparseIdx_t scalarIntZero = 0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.row_indices, &scalarIntZero, sizeof(clsparseIdx_t), 0,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.row_indices");


        CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.col_indices, &scalarIntZero, sizeof(clsparseIdx_t), 0,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.col_indices");

	}// end

	void reset_gpu_write_buffer()
	{
		T scalar = 0;
		CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.values, &scalar, sizeof(T), 0,
			cooMtx.num_nonzeros * sizeof(T), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.values");

        clsparseIdx_t scalarIntZero = 0;
        CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.row_indices, &scalarIntZero, sizeof(clsparseIdx_t), 0,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.row_indices");


        CLSPARSE_V(::clEnqueueFillBuffer(queue, cooMtx.col_indices, &scalarIntZero, sizeof(clsparseIdx_t), 0,
            cooMtx.num_nonzeros * sizeof(clsparseIdx_t), 0, NULL, NULL), "::clEnqueueFillBuffer cooMtx.col_indices");
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
			//Host to GPU: CSR-> [row_pointer(num_rows + 1) + Column Indices] * sizeof(int) + sizeof(T) * (num_nonzero)
			//GPU to Host: Coo - > row_indices + Col_indices + Values- > [sizeof(T) * num_nonzero] + sizeof(int)
			size_t sparseBytes = sizeof(cl_int) * (csrMtx.num_nonzeros + csrMtx.num_rows + 1) + sizeof(T) * (csrMtx.num_nonzeros) +
				          sizeof(T) * (cooMtx.num_nonzeros) + sizeof(cl_int) * (cooMtx.num_nonzeros * 2);
			cpuTimer->pruneOutliers(3.0);
			cpuTimer->Print(sparseBytes, "GiB/s");
			cpuTimer->Reset();

			gpuTimer->pruneOutliers(3.0);
			gpuTimer->Print(sparseBytes, "GiB/s");
			gpuTimer->Reset();
#endif
			// Calculate Number of Elements transformed per unit time
            clsparseIdx_t sparseElements = csrMtx.num_nonzeros;
			cpuTimer->pruneOutliers(3.0);
			cpuTimer->Print(sparseElements, "GiElements/s");
			cpuTimer->Reset();

			gpuTimer->pruneOutliers(3.0);
			gpuTimer->Print(sparseElements, "GiElements/s");
			gpuTimer->Reset();
		}

		//this is necessary since we are running a iteration of tests and calculate the average time. (in client.cpp)
		//need to do this before we eventually hit the destructor
        clsparseCsrMetaDelete( &csrMtx );
        CLSPARSE_V(::clReleaseMemObject(csrMtx.values), "clReleaseMemObject csrMtx.values");
		CLSPARSE_V(::clReleaseMemObject(csrMtx.col_indices), "clReleaseMemObject csrMtx.col_indices");
		CLSPARSE_V(::clReleaseMemObject(csrMtx.row_pointer), "clReleaseMemObject csrMtx.row_pointer");

		CLSPARSE_V(::clReleaseMemObject(cooMtx.values), "clReleaseMemObject cooMtx.values");
		CLSPARSE_V(::clReleaseMemObject(cooMtx.col_indices), "clReleaseMemObject cooMtx.col_indices");
		CLSPARSE_V(::clReleaseMemObject(cooMtx.row_indices), "clReleaseMemObject cooMtx.row_indices");
	}

private:
	void xCsr2Coo_Function(bool flush);

	// Timers
	clsparseTimer* gpuTimer;
	clsparseTimer* cpuTimer;
	size_t gpuTimerID;
	size_t cpuTimerID;

	std::string sparseFile;

	//device values
	clsparseCsrMatrix csrMtx;
	clsparseCooMatrix cooMtx;

        // host values
        cl_bool explicit_zeroes;

	//OpenCL state
	cl_command_queue_properties cqProp;

}; // class xCsr2Coo

template<> void
xCsr2Coo<float>::xCsr2Coo_Function(bool flush)
{
	clsparseStatus status = clsparseScsr2coo(&csrMtx, &cooMtx, control);
	if (flush)
		clFinish(queue);
}// end

template<> void
xCsr2Coo<double>::xCsr2Coo_Function(bool flush)
{
	clsparseStatus status = clsparseDcsr2coo(&csrMtx, &cooMtx, control);
	if (flush)
		clFinish(queue);
}// end


#endif // ifndef CLSPARSE_BENCHMARK_xCsr2Coo_HXX
