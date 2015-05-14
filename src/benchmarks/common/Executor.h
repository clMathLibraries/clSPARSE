#ifndef __EXECUTOR_H_
#define __EXECUTOR_H_


#include "clSPARSE.h"

#include "opencl_utils.h"

#include "Params.h"
#include "Find.h"
#include "MatrixUtils.h"
#include "MatrixStatistics.h"
#include "Timer.h"

namespace fs = boost::filesystem;


template<typename T>
class Executor
{

public:

    Executor(const Params<T>& params)
    {
        this->params = params;

        cl_int status = CL_SUCCESS;

        cl::Device device = getDevice(params.pID, params.dID);

        std::cout << "Using device " << device.getInfo<CL_DEVICE_NAME>() << std::endl;

        context = clCreateContext(NULL, 1, &device(), NULL, NULL, NULL);
        queue = clCreateCommandQueue(context, device(), 0, NULL);

        clsparseSetup();
        control = clsparseCreateControl(queue, NULL);
        clsparseEnableAsync(control, true);
    }

    ~Executor()
    {
        clsparseReleaseControl(control);
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
            std::string path = (*file).string( );

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
                std::cerr << "Problem with reading matrix "
                          << (*file).filename( ) << std::endl;
                continue; //we did not allocated cl_mem so it should be ok
            }

            cl_int status;

            //gpu matrix definition
            clsparseCsrMatrix csrMatx;
            clsparseInitCsrMatrix(&csrMatx);


            if (!allocateMatrix(row_offsets, col_indices, values, n_rows, n_cols,
                                csrMatx, queue, context, &status))
            {
                std::cerr << "Problem with allocating matrix "
                    << ( *file ).filename( )
                          <<  " OpenCL status " << status << std::endl;
                return -102;
            }

            //TODO:: remember about offsets;
            clsparseScalar cl_alpha;
            clsparseInitScalar(&cl_alpha);
            clsparseScalar cl_beta;
            clsparseInitScalar(&cl_beta);

            clsparseVector cl_x;
            clsparseInitVector(&cl_x);
            clsparseVector cl_y;
            clsparseInitVector(&cl_y);

            if (!allocateScalar<T>(cl_alpha, 1, params.alpha, queue, context, &status))
            {
                std::cerr << "Problem with allocation alpha " << status << std::endl;
                return status;
            }

            if (!allocateScalar<T>(cl_beta, 1 , params.beta, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }

            if (!allocateVector<T>(cl_x, n_cols , 1, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }

            if (!allocateVector<T>(cl_y, n_rows , 0, queue, context, &status))
            {
                std::cerr << "Problem with allocation beta " << status << std::endl;
                return status;
            }


            bool warmup_status =
                    executeCSRMultiply<T>(cl_alpha, csrMatx, cl_x, cl_beta, cl_y,
                                          control, params.number_of_warmups);
            if (!warmup_status)
            {
                std::cerr << "Problem with multiply during warmup" << std::endl;
                std::cerr << " Problematic matrix: " <<
                             (*file).filename( ) << std::endl;
                return -103;
            }

            CPerfCounter timer;
            timer.Start();
            bool bench_status =
                    executeCSRMultiply<T>(cl_alpha, csrMatx, cl_x, cl_beta, cl_y,
                                          control, params.number_of_warmups);
            timer.Stop();
            if (! bench_status)
            {
                std::cerr << "Problem with multiply during bench" << std::endl;
                std::cerr << " Problematic matrix: " <<
                             (*file).filename( ) << std::endl;
                return -104;
            }

            double average_time = timer.GetElapsedTime() / params.number_of_tries;
            double instr_bandwidth = 1e-9 * (2*n_vals) / average_time;
            double memory_bandwidth = 1e-9 * (n_vals * (2 * sizeof(T) + sizeof(int)) +
                                      n_rows * (sizeof(T) + sizeof(int)));
            memory_bandwidth /= average_time;

            MatrixStatistics m = { n_rows, n_cols, n_vals, n_vals/n_rows,
                                   average_time,
                                   instr_bandwidth,
                                   memory_bandwidth,
                                   (*file).filename( ).string( ) };
            results.push_back(m);

            //release resources;
            clReleaseMemObject(csrMatx.colIndices);
            clReleaseMemObject(csrMatx.rowOffsets);
            clReleaseMemObject(csrMatx.values);
            clReleaseMemObject(cl_x.values);
            clReleaseMemObject(cl_y.values);
            clReleaseMemObject(cl_alpha.value);
            clReleaseMemObject(cl_beta.value);


        }

        printMatrixStatistics(results);

    }

private:


    Params<T> params;

    cl_context context;
    cl_command_queue queue;
    clsparseControl control;

    std::vector<fs::path> matrix_files;
    std::vector<MatrixStatistics> results;




};


#endif //__EXECUTOR