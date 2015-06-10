#include <iostream>
#include <vector>

#include <CL/cl.hpp>

#include <clSPARSE.h>

/**
 * Sample Conjugate Gradients Solver (CG C++)
 * Solves equation A * x = b
 *
 * A - [m x n] matrix in CSR format
 * x - dense vector of n elements (unknowns)
 * b - dense vector of m elements (rhs)
 *
 * Program presents usage of clSPARSE Conjugate Gradients iterative algorithm
 * for solving system of linear equations with positive definite matrix in CSR
 * format.
 *
 * Currently clSPARSE offers only one preconditioner which is Jacobi (Diagonal)
 * algorithm. (check src/solvers/preconditioners/preconditioner.hpp and diagonal.hpp)
 *
 * For more theoretical details check
 * http://www.cs.cmu.edu/~./quake-papers/painless-conjugate-gradient.pdf
 *
 * */

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

    std::cout << "Executing sample clSPARSE CG Solver (A*x = b) C++" << std::endl;

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

    clsparseVector x;
    clsparseInitVector(&x);

    clsparseVector b;
    clsparseInitVector(&b);

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

    A.nnz = nnz;
    A.m = row;
    A.n = col;

    // This function allocates memory for rowBlocks structure. If not called
    // the structure will not be calculated and clSPARSE will run the vectorized
    // version of SpMV instead of adaptive;
    clsparseCsrMetaSize( &A, control );

    // Allocate memory for CSR matrix
    A.values = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                 A.nnz * sizeof( float ), NULL, &cl_status );

    A.colIndices = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     A.nnz * sizeof( cl_int ), NULL, &cl_status );

    A.rowOffsets = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                     ( A.m + 1 ) * sizeof( cl_int ), NULL, &cl_status );

    A.rowBlocks = ::clCreateBuffer( context(), CL_MEM_READ_ONLY,
                                    A.rowBlockSize * sizeof( cl_ulong ), NULL, &cl_status );


    fileError = clsparseSCsrMatrixfromFile( &A, matrix_path.c_str( ), control );


    if (fileError != clsparseSuccess)
    {
        std::cout << "Problem with reading matrix from " << matrix_path
                  << " Error: " << status << std::endl;
        return -6;
    }

    float one = 1.0f;
    float zero = 0.0f;

    // Allocate memory for vector of unknowns;
    // We will fill it with zeros as a initial guess
    x.n = A.n;
    x.values = clCreateBuffer(context(), CL_MEM_READ_ONLY, x.n * sizeof(float),
                              NULL, &cl_status);

    cl_status = clEnqueueFillBuffer(queue(), x.values, &zero, sizeof(float),
                                    0, x.n * sizeof(float), 0, nullptr, nullptr);

    // Allocate memory for rhs vector

    b.n = A.m;
    b.values = clCreateBuffer(context(), CL_MEM_READ_WRITE, b.n * sizeof(float),
                              NULL, &cl_status);
    // Fill it with ones.
    cl_status = clEnqueueFillBuffer(queue(), b.values, &one, sizeof(float),
                                    0, b.n * sizeof(float), 0, nullptr, nullptr);



    /**Step 4. Call the conjugate gradients algorithm */

    // Create solver control object. It keeps the informations
    // about the used preconditioner, desired relative and absolute tolerances
    // and maximal number of iterations to be performed;

    /* TODO:: missing getters of properties for solver control */

    // We will use:
    // preconditioner: diagonal
    // relative tolerance: 1e-2
    // absolute tolerance: 1e-5
    // max iters: 1000
    clSParseSolverControl solverControl =
            clsparseCreateSolverControl(DIAGONAL, 1000, 1e-2, 1e-5);

    // We can set different print modes of the solver status:
    // QUIET - print no messages (default)
    // NORMAL - print summary
    // VERBOSE - per iteration status;
    clsparseSolverPrintMode(solverControl, VERBOSE);

    /* TODO: provide various solver statuses for different scenarios
     * Solver reached max numer of iterations is not a failure.
     */

    status = clsparseScsrcg(&x, &A, &b, solverControl, control);

    //release solver control structure after finishing execution;
    clsparseReleaseSolverControl(solverControl);



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
    clReleaseMemObject ( b.values );

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}
