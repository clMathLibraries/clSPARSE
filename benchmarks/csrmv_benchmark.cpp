#include <iostream>
#include <vector>

#include "clSPARSE.h"
#include "Timer.h"
#include "Find.h"
#include "MatrixUtils.h"

#include "opencl_utils.h"
#include "matrix_market.h"

namespace fs = boost::filesystem;

int main(int argc, char* argv[])
{

    CPerfCounter timer;
    timer.Start();

    cl_int status = CL_SUCCESS;

    cl_platform_id* platforms = NULL;
    cl_uint num_platforms = 0;

    status = getPlatforms(&platforms, &num_platforms);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Problem with setting up the OpneCL platforms" << std::endl;
        return status;
    }

    printPlatforms(platforms, num_platforms);

    cl_device_id device = NULL;
    status = getDevice(platforms[0], &device, CL_DEVICE_TYPE_GPU);
    if (status != CL_SUCCESS)
    {
        std::cerr << "Problem with initializing GPU device" << std::endl;
        return status;
    }

    printDeviceInfo(device);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    clsparseSetup();
    free(platforms);


    std::vector<fs::path> matrix_files;

    if (!findMatrices("/media/jpola/Storage/matrices/small_set", "mtx", matrix_files))
        return -100;

    auto file = matrix_files.begin();

    std::string path = (*file).native();

    //host matrix definition
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<float> values;
    int n_rows = 0;
    int n_cols = 0;
    int n_vals = 0;

    if (!readMatrixMarketCSR(row_offsets, col_indices, values,
                        n_rows, n_cols, n_vals, path))
    {
        return -101;
    }

    //gpu matrix definition
    cl_mem cl_row_offsets;
    cl_mem cl_col_indices;
    cl_mem cl_values;

    if (!allocateMatrix(row_offsets, col_indices, values,
                   n_rows, n_cols,
                   &cl_row_offsets, &cl_col_indices, &cl_values,
                   queue, context, &status))
    {
        std::cerr << "Problem with allocating matrix "
                  << (*file).filename().native()
                  <<  " OpenCL status " << status << std::endl;
        return status;
    }


    //TODO:: remember about offsets;
    cl_mem alpha;
    cl_mem beta;

    cl_mem x;
    cl_mem y;

    if (!allocateVector<float>(&alpha, 1, 1, queue, context, &status))
    {
        std::cerr << "Problem with allocation alpha " << status << std::endl;
        return status;
    }

    if (!allocateVector<float>(&beta, 1 , 1, queue, context, &status))
    {
        std::cerr << "Problem with allocation beta " << status << std::endl;
        return status;
    }

    if (!allocateVector<float>(&x, n_cols , 1, queue, context, &status))
    {
        std::cerr << "Problem with allocation beta " << status << std::endl;
        return status;
    }

    if (!allocateVector<float>(&y, n_rows , 0, queue, context, &status))
    {
        std::cerr << "Problem with allocation beta " << status << std::endl;
        return status;
    }

    executeCSRMultiply<float>(n_rows, n_cols, n_vals,
                       alpha, 0,
                       cl_row_offsets,
                       cl_col_indices,
                       cl_values,
                       x,
                       0,
                       beta,
                       0,
                       y,
                       0,
                       queue,
                       1
                       );



    clsparseTeardown();
    timer.Stop();
    std::cout << "Execution time "
              << timer.GetElapsedTime() << " s." << std::endl;
	return 0;
}
