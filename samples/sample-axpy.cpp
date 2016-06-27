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

/*! \file
 * \brief Simple demonstration code for how to execute an Axpy operation with
 * clSPARSE
 */

#include <iostream>
#include <vector>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION
#include <CL/cl2.hpp>

#include "clSPARSE.h"
#include "clSPARSE-error.h"

/*!
 * \brief Sample AXPY (C++)
 * \details Program presents use of clSPARSE library in AXPY (y = alpha * x + y) operation
 * by completing following steps:
 * 1. Setup OpenCL environment
 * 2. Setup GPU buffers
 * 3. Init clSPARSE library
 * 4. Execute algorithm cldenseSaxpy
 * 5. Shutdown clSPARSE library & OpenCL
 *
 * \note UNIX Hint: Before allocating more than 3GB of VRAM define GPU_FORCE_64BIT_PTR=1
 * in your system environment to enable 64bit addresing;
 */

int main(int argc, char* argv[])
{
    std::cout << "Executing sample clSPARSE AXPY (y = alpha * x + y) C++" << std::endl;

/**  Step 1. Setup OpenCL environment; **/

    // Init OpenCL environment;
    cl_int cl_status;

    // Get OpenCL platforms
    std::vector<cl::Platform> platforms;

    cl_status = cl::Platform::get(&platforms);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting OpenCL platforms"
                  << " [" << cl_status << "]" << std::endl;
        return -1;
    }

    int platform_id = 0;
    for (const auto& p : platforms)
    {
        std::cout << "Platform ID " << platform_id++ << " : "
                  << p.getInfo<CL_PLATFORM_NAME>() << std::endl;

    }

    // Using first platform
    platform_id = 0;
    cl::Platform platform = platforms[platform_id];

    // Get device from platform
    std::vector<cl::Device> devices;
    cl_status = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting devices from platform"
                  << " [" << platform_id << "] " << platform.getInfo<CL_PLATFORM_NAME>()
                  << " error: [" << cl_status << "]" << std::endl;
    }

    std::cout << std::endl
              << "Getting devices from platform " << platform_id << std::endl;
    cl_int device_id = 0;
    for (const auto& device : devices)
    {
        std::cout << "Device ID " << device_id++ << " : "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;


    }

    // Using first device;
    device_id = 0;
    cl::Device device = devices[device_id];

    // Create OpenCL context;
    cl::Context context (device);

    // Create OpenCL queue;
    cl::CommandQueue queue(context, device);

/**  Step 2. Setup GPU buffers **/

    // Let's create host buffers first.

    // size of the vectors;
    int N = 1024;

    float alpha = 2.0f;

    std::vector<float> y(N, 2.0f);
    std::vector<float> x(N, 1.0f);

    // GPU buffers
    clsparseScalar gpuAlpha;
    clsparseInitScalar(&gpuAlpha);

    cldenseVector gpuY;
    clsparseInitVector(&gpuY);

    cldenseVector gpuX;
    clsparseInitVector(&gpuX);


    // Allocate alpha on gpu by using host pointer
    gpuAlpha.value = ::clCreateBuffer (context(), CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                          sizeof(float), &alpha, &cl_status);
    if (cl_status != CL_SUCCESS )
    {
        std::cout << "Problem with allocating alpha buffer on GPU\n " << std::endl;
    }


    gpuY.values = ::clCreateBuffer (context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                N * sizeof(float), y.data(), &cl_status);

    // set the size of cldenseVector;
    gpuY.num_values = N;

    if (cl_status != CL_SUCCESS )
    {
        std::cout << "Problem with allocating Y buffer on GPU\n " << std::endl;
    }

   gpuX.values = ::clCreateBuffer (context(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                                N * sizeof(float), x.data(), &cl_status);
   // set the size of cldenseVector;
   gpuX.num_values = N;

    if (cl_status != CL_SUCCESS )
    {
        std::cout << "Problem with allocating X buffer on GPU\n " << std::endl;
    }


    /** Step 3. Init clSPARSE library **/

	clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
        return -1;
    }


    // Create clsparseControl object
    clsparseCreateResult createResult = clsparseCreateControl( queue( ) );
    CLSPARSE_V( createResult.status, "Failed to create clsparse control" );
    /** Step 4. Execute AXPY algorithm **/

    status = cldenseSaxpy(&gpuY, &gpuAlpha, &gpuX, &gpuY, createResult.control );

    if (status != clsparseSuccess)
    {
        std::cout << "Problem with execution of clsparse AXPY algorithm"
                  << " error: [" << status << "]" << std::endl;
    }

    /** Step 5. Shutdown clSPARSE library & OpenCL **/
    status = clsparseReleaseControl( createResult.control );

	status = clsparseTeardown();
    if (status != clsparseSuccess)
    {
        std::cerr << "Problem with executing clsparseTeardown()" << std::endl;
        return -2;
    }

    // Get results back to the host
    ::clEnqueueReadBuffer(queue(), gpuY.values, CL_TRUE, 0, N * sizeof(float),
                          y.data(), 0, nullptr, nullptr);

    std::cout << "Result data: " << std::endl;
    for ( int i = 0; i < 5; i++)
    {
        std::cout << "\t" << i << " = " << y[i] << std::endl;
    }


    ::clReleaseMemObject(gpuAlpha.value);
    ::clReleaseMemObject(gpuY.values);
    gpuY.num_values = 0;
    ::clReleaseMemObject(gpuX.values);
    gpuX.num_values = 0;

    //OpenCL Wrapper automatically release allocated resources

    std::cout << "Program completed" << std::endl;
	return 0;
}
