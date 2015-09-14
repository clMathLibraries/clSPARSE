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
 * \brief Simple demonstration code for how to calculate a SpM-dV (Sparse matrix
 * times dense Vector) multiply
 */

#include <iostream>
#include <vector>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

#include <clSPARSE.h>

/**
 * \brief Sample Sparse Matrix dense Vector multiplication (SPMV C++)
 *  \details [y = alpha * A*x + beta * y]
 *
 * A - [m x n] matrix in CSR format
 * x - dense vector of n elements
 * y - dense vector of m elements
 * alpha, beta - scalars
 *
 *
 * Program presents usage of clSPARSE library in csrmv (y = A*x) operation
 * where A is sparse matrix in CSR format, x, y are dense vectors.
 *
 * clSPARSE offers two spmv algorithms for matrix stored in CSR format.
 * First one is called vector algorithm, the second one is called adaptve.
 * Adaptive version is usually faster but for given matrix additional
 * structure (rowBlocks) needs to be calculated first.
 *
 * To calculate rowBlock structure you have to execute clsparseCsrMetaSize
 * for given matrix stored in CSR format. It is enough to calculate the
 * structure once, it is related to its nnz pattern.
 *
 * After the matrix is read from disk with the function
 * clsparse<S,D>CsrMatrixfromFile
 * the rowBlock structure can be calculated using clsparseCsrMetaCompute
 *
 * If rowBlocks are calculated the clsparseCsrMatrix.rowBlocks field is not null.
 *
 * Program is executing by completing following steps:
 * 1. Setup OpenCL environment
 * 2. Setup GPU buffers
 * 3. Init clSPARSE library
 * 4. Execute algorithm cldenseSaxpy
 * 5. Shutdown clSPARSE library & OpenCL
 *
 * usage:
 *
 * sample-spmv path/to/matrix/in/mtx/format.mtx
 *
 */

int main (int argc, char* argv[])
{
    //parse command line
    std::string matrix_path;

    if (argc < 2)
    {
        std::cout << "Not enough parameters. "
                  << "Please specify path to matrix in mtx format as parameter"
                  << std::endl;
        return -1;
    }
    else
    {
        matrix_path = std::string(argv[1]);
    }

    std::cout << "Executing sample clSPARSE SpMV (y = A*x) C++" << std::endl;

    std::cout << "Matrix will be read from: " << matrix_path << std::endl;

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
        return -2;
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

    /** Step 2. Setup GPU buffers **/

    //we will allocate it after matrix will be loaded;
    clsparseScalar alpha;
    clsparseInitScalar(&alpha);

    alpha.value = clCreateBuffer(context(), CL_MEM_READ_ONLY, sizeof(float),
                                 nullptr, &cl_status);

    clsparseScalar beta;
    clsparseInitScalar(&beta);

    beta.value = clCreateBuffer(context(), CL_MEM_READ_ONLY, sizeof(float),
                                nullptr, &cl_status);


    cldenseVector x;
    clsparseInitVector(&x);

    cldenseVector y;
    clsparseInitVector(&y);

    clsparseCsrMatrix A;
    clsparseInitCsrMatrix(&A);

    /** Step 3. Init clSPARSE library **/

    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
        return -3;
    }


    // Create clsparseControl object
    clsparseControl control = clsparseCreateControl(queue(), &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with creating clSPARSE control object"
                  <<" error [" << status << "]" << std::endl;
        return -4;
    }


    // Read matrix from file. Calculates the rowBlocks strucutres as well.
    int nnz, row, col;
    // read MM header to get the size of the matrix;
    clsparseStatus fileError
            = clsparseHeaderfromFile( &nnz, &row, &col, matrix_path.c_str( ) );

    if( fileError != clsparseSuccess )
    {
        std::cout << "Could not read matrix market header from disk" << std::endl;
        return -5;
    }

    A.num_nonzeros = nnz;
    A.num_rows = row;
    A.num_cols = col;

    // Allocate memory for CSR matrix
    A.values = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                 A.num_nonzeros * sizeof( float ), NULL, &cl_status );

    A.colIndices = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     A.num_nonzeros * sizeof( cl_int ), NULL, &cl_status );

    A.rowOffsets = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     ( A.num_rows + 1 ) * sizeof( cl_int ), NULL, &cl_status );

    A.rowBlocks = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                    A.rowBlockSize * sizeof( cl_ulong ), NULL, &cl_status );


    fileError = clsparseSCsrMatrixfromFile( &A, matrix_path.c_str( ), control );

    // This function allocates memory for rowBlocks structure. If not called
    // the structure will not be calculated and clSPARSE will run the vectorized
    // version of SpMV instead of adaptive;
    clsparseCsrMetaSize( &A, control );
    A.rowBlocks = ::clCreateBuffer( context(), CL_MEM_READ_WRITE,
            A.rowBlockSize * sizeof( cl_ulong ), NULL, &cl_status );
    clsparseCsrMetaCompute( &A, control );

    if (fileError != clsparseSuccess)
    {
        std::cout << "Problem with reading matrix from " << matrix_path
                  << " Error: " << status << std::endl;
        return -6;
    }

    float one = 1.0f;
    float zero = 0.0f;

    // alpha = 1;
    float* halpha = (float*) clEnqueueMapBuffer(queue(), alpha.value, CL_TRUE, CL_MAP_WRITE,
                                                0, sizeof(float), 0, nullptr, nullptr, &cl_status);
    *halpha = one;

    cl_status = clEnqueueUnmapMemObject(queue(), alpha.value, halpha,
                                        0, nullptr, nullptr);

    //beta = 0;
    float* hbeta = (float*) clEnqueueMapBuffer(queue(), beta.value, CL_TRUE, CL_MAP_WRITE,
                                               0, sizeof(float), 0, nullptr, nullptr, &cl_status);
    *hbeta = zero;

    cl_status = clEnqueueUnmapMemObject(queue(), beta.value, hbeta,
                                        0, nullptr, nullptr);


    x.num_values = A.num_cols;
    x.values = clCreateBuffer(context(), CL_MEM_READ_ONLY, x.num_values * sizeof(float),
                              NULL, &cl_status);

    cl_status = clEnqueueFillBuffer(queue(), x.values, &one, sizeof(float),
                                    0, x.num_values * sizeof(float), 0, nullptr, nullptr);

    y.num_values = A.num_rows;
    y.values = clCreateBuffer(context(), CL_MEM_READ_WRITE, y.num_values * sizeof(float),
                              NULL, &cl_status);

    cl_status = clEnqueueFillBuffer(queue(), y.values, &zero, sizeof(float),
                                    0, y.num_values * sizeof(float), 0, nullptr, nullptr);




    /**Step 4. Call the spmv algorithm */
    status = clsparseScsrmv(&alpha, &A, &x, &beta, &y, control);

    if (status != clsparseSuccess)
    {
        std::cout << "Problem with execution SpMV algorithm."
                  << " Error: " << status << std::endl;
    }


    /** Step 5. Close & release resources */
    status = clsparseReleaseControl(control);
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with releasing control object."
                  << " Error: " << status << std::endl;
    }

    status = clsparseTeardown();

    if (status != clsparseSuccess)
    {
        std::cout << "Problem with closing clSPARSE library."
                  << " Error: " << status << std::endl;
    }


    //release mem;
    clReleaseMemObject ( A.values );
    clReleaseMemObject ( A.colIndices );
    clReleaseMemObject ( A.rowOffsets );
    clReleaseMemObject ( A.rowBlocks );

    clReleaseMemObject ( x.values );
    clReleaseMemObject ( y.values );

    clReleaseMemObject ( alpha.value );
    clReleaseMemObject ( beta.value );

    std::cout << "Program completed successfully." << std::endl;

    return 0;
}
