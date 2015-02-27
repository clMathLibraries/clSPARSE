#ifndef __EXECUTOR_H_
#define __EXECUTOR_H_

#include "clSPARSE.h"

#include "Params.h"
#include "Find.h"
#include "MatrixUtils.h"

namespace fs = boost::filesystem;


template<typename T>
class Executor
{


public:

    Executor(const Params<T>& params)
    {
        this->params = params;

        cl_int status = CL_SUCCESS;

        cl_platform_id* platforms = NULL;
        cl_uint num_platforms = 0;

        status = getPlatforms(&platforms, &num_platforms);
        if (status != CL_SUCCESS)
        {
            std::cerr << "Problem with setting up the OpneCL platforms" << std::endl;
        }

        printPlatforms(platforms, num_platforms);

        cl_device_id device = NULL;
        status = getDevice(platforms[0], &device, CL_DEVICE_TYPE_GPU);
        if (status != CL_SUCCESS)
        {
            std::cerr << "Problem with initializing GPU device" << std::endl;
        }

        printDeviceInfo(device);

        context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device, 0, NULL);

        clsparseSetup();
        free(platforms);
    }

    ~Executor()
    {
      clsparseTeardown();
      clReleaseCommandQueue(queue);
      clReleaseContext(context);

    }

    cl_int exec()
    {
        if (!findMatrices(params.root_dir, "mtx", matrix_files))
            return -100;

        for ( auto file = matrix_files.begin();
              file != matrix_files.end(); file++)
        {
            std::string path = (*file).native();

            //host matrix definition
            std::vector<int> row_offsets;
            std::vector<int> col_indices;
            std::vector<T> values;
            int n_rows = 0;
            int n_cols = 0;
            int n_vals = 0;

            if (!readMatrixMarketCSR(row_offsets, col_indices, values,
                                     n_rows, n_cols, n_vals, path))
            {
                return -101;
            }

            cl_int status;
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
                return -102;
            }

            //TODO:: remember about offsets;
            cl_mem cl_alpha;
            cl_mem cl_beta;

            cl_mem cl_x;
            cl_mem cl_y;

            if (!allocateVector<T>(&cl_alpha, 1, params.alpha, queue, context, &status))
            {
                std::cerr << "Problem with allocation alpha " << status << std::endl;
                return status;
            }

            if (!allocateVector<T>(&cl_beta, 1 , params.beta, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }

            if (!allocateVector<T>(&cl_x, n_cols , 1, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }

            if (!allocateVector<T>(&cl_y, n_rows , 0, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }


            bool warmup_status =
                    executeCSRMultiply<T>(n_rows, n_cols, n_vals,
                                  cl_alpha, 0,
                                  cl_row_offsets, cl_col_indices, cl_values,
                                  cl_x, 0,
                                  cl_beta, 0,
                                  cl_y, 0,
                                  queue,
                                  params.number_of_warmups);
            if (!warmup_status)
            {
                std::cerr << "Problem with multiply during warmup" << std::endl;
                std::cerr << " Problematic matrix: " <<
                             (*file).filename().native() << std::endl;
                return -103;
            }


            bool bench_status =
                    executeCSRMultiply<T>(n_rows, n_cols, n_vals,
                                  cl_alpha, 0,
                                  cl_row_offsets, cl_col_indices, cl_values,
                                  cl_x, 0,
                                  cl_beta, 0,
                                  cl_y, 0,
                                  queue,
                                  params.number_of_tries);
            if (! bench_status)
            {
                std::cerr << "Problem with multiply during bench" << std::endl;
                std::cerr << " Problematic matrix: " <<
                             (*file).filename().native() << std::endl;
                return -104;
           }

            //release resources;
            clReleaseMemObject(cl_row_offsets);
            clReleaseMemObject(cl_col_indices);
            clReleaseMemObject(cl_values);
            clReleaseMemObject(cl_x);
            clReleaseMemObject(cl_y);
            clReleaseMemObject(cl_alpha);
            clReleaseMemObject(cl_beta);


        }



    }

private:


    Params<T> params;

    cl_context context;
    cl_command_queue queue;

    std::vector<fs::path> matrix_files;





};


#endif //__EXECUTOR
