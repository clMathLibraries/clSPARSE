#ifndef _CSR_MATRIX_ENVIRONMENT_H_
#define _CSR_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "matrix_market.h"


/**
 * @brief The CSREnvironment class will keep the input parameters for tests
 * They are list of paths to matrices in csr format in mtx files.
 */
class CSREnvironment : public ::testing::Environment
{
public:
    explicit CSREnvironment(const std::string& command_line_arg,
                            cl_command_queue queue,
                            cl_context context)
    {
        this->context = context;
        this->queue = queue;

        file_name = command_line_arg;
    }


    //read the matrix file passed by command_line_arg and allocate cl_buffers;
    void SetUp()
    {
        bool read_status = false;
        read_status = readMatrixMarketCSR(row_offsets, col_indices, f_values,
                                          n_rows, n_cols, n_vals, file_name);
        if(!read_status)
        {
            exit(-3);
        }

        d_values = std::vector<double>(f_values.begin(), f_values.end());

        //ALLOCATE MATRIX CL Buffers;
        cl_int mem_status;

        cl_row_offsets = clCreateBuffer(context, CL_MEM_READ_WRITE, row_offsets.size()*sizeof(int), NULL, &mem_status);
        cl_col_indices = clCreateBuffer(context, CL_MEM_READ_WRITE, col_indices.size()*sizeof(int), NULL, &mem_status);
        cl_f_values = clCreateBuffer(context, CL_MEM_READ_WRITE, f_values.size()*sizeof(float), NULL, &mem_status);
        cl_d_values = clCreateBuffer(context, CL_MEM_READ_WRITE, d_values.size()*sizeof(double), NULL, &mem_status);

        if(mem_status != CL_SUCCESS)
        {
            TearDown();
            exit(-4);
        }

        cl_int copy_status;
        // copy host matrix arrays to device;
        copy_status = clEnqueueWriteBuffer(queue, cl_row_offsets, 1, 0,
                                           row_offsets.size() * sizeof(int),
                                           row_offsets.data(),
                                           0, NULL, NULL);

        copy_status = clEnqueueWriteBuffer(queue, cl_col_indices, 1, 0,
                                           col_indices.size() * sizeof(int),
                                           col_indices.data(),
                                           0, NULL, NULL);

        copy_status = clEnqueueWriteBuffer(queue, cl_f_values, 1, 0,
                                           f_values.size() * sizeof(cl_float),
                                           f_values.data(),
                                           0, NULL, NULL);
        copy_status = clEnqueueWriteBuffer(queue, cl_d_values, 1, 0,
                                           d_values.size() * sizeof(cl_double),
                                           d_values.data(),
                                           0, NULL, NULL);

        if(copy_status != CL_SUCCESS)
        {
            TearDown();
            exit(-5);
        }
    }

    //cleanup
    void TearDown()
    {
        //release buffers;
        clReleaseMemObject(cl_row_offsets);
        clReleaseMemObject(cl_col_indices);
        clReleaseMemObject(cl_f_values);
        clReleaseMemObject(cl_d_values);

    }

    static std::vector<int> row_offsets;
    static std::vector<int> col_indices;
    static std::vector<float> f_values;
    static std::vector<double> d_values;
    static int n_rows, n_cols, n_vals;

    //cl buffers for above matrix definition;
    static cl_mem cl_row_offsets;
    static cl_mem cl_col_indices;
    static cl_mem cl_f_values;
    static cl_mem cl_d_values;

private :
    cl_command_queue queue;
    cl_context context;
    std::string file_name;
};

#endif //_CSR_MATRIX_ENVIRONMENT_H_
