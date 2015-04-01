#ifndef __MATRIX_UTILS_H__
#define __MATRIX_UTILS_H__

#include <boost/filesystem.hpp>
#include <string>
#include <vector>

#include "matrix_market.h"
#include "clSPARSE.h"

template<typename INDEX_TYPE, typename VALUE_TYPE>
bool allocateMatrix(const std::vector<INDEX_TYPE>& rows,
                    const std::vector<INDEX_TYPE>& cols,
                    const std::vector<VALUE_TYPE>& values,
                    const INDEX_TYPE n_rows,
                    const INDEX_TYPE n_cols,
                    cl_mem* cl_rows,
                    cl_mem* cl_cols,
                    cl_mem* cl_vals,
                    cl_command_queue queue,
                    cl_context context,
                    cl_int* status)
{
    const std::type_info& type_desc = typeid(VALUE_TYPE);


    if (typeid(VALUE_TYPE) != typeid(cl_float)
            && typeid(VALUE_TYPE) != typeid(cl_double))
    {
        std::cout << "Not implemented for this type "
                  << type_desc.name() <<std::endl;
        return false;
    }
    cl_int mem_status;

    // create matrix buffers
    *cl_rows = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    rows.size()*sizeof(INDEX_TYPE),
                                    NULL, &mem_status);
    *cl_cols = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    cols.size()*sizeof(INDEX_TYPE),
                                    NULL, &mem_status);
    *cl_vals = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               values.size()*sizeof(VALUE_TYPE),
                               NULL, &mem_status);



    // copy host matrix arrays to device;
    mem_status |= clEnqueueWriteBuffer(queue, *cl_rows, 1, 0,
                                       rows.size() * sizeof(INDEX_TYPE),
                                       rows.data(),
                                       0, NULL, NULL);
    mem_status |= clEnqueueWriteBuffer(queue, *cl_cols, 1, 0,
                                       cols.size() * sizeof(INDEX_TYPE),
                                       cols.data(),
                                       0, NULL, NULL);
    mem_status |= clEnqueueWriteBuffer(queue, *cl_vals, 1, 0,
                                       values.size() * sizeof(VALUE_TYPE),
                                       values.data(),
                                       0, NULL, NULL);

    if (mem_status != CL_SUCCESS)
    {
        clReleaseMemObject(*cl_rows);
        clReleaseMemObject(*cl_cols);
        clReleaseMemObject(*cl_vals);
        *status = mem_status;
        return false;
    }

    *status = mem_status;
    return true;
}

template<typename VALUE_TYPE>
bool allocateVector(cl_mem* buffer, size_t size, VALUE_TYPE value,
                    cl_command_queue queue,
                    cl_context context,
                    cl_int* status)
{
    //fix for CL 1.1 support (Nvidia devices)
    {
        std::vector<VALUE_TYPE> fill_vec(size, value);
        *buffer = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         size * sizeof(VALUE_TYPE),
                                         fill_vec.data(), status);

    }
//    *buffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
//                                 size * sizeof(VALUE_TYPE),
//                                 NULL, status);
//    if (*status != CL_SUCCESS)
//    {
//        return false;
//    }

//    cl_event fill_event;

//    *status = clEnqueueFillBuffer(queue, *buffer, &value, sizeof(VALUE_TYPE),
//                        0, size * sizeof(VALUE_TYPE), 0, NULL, &fill_event);

//    if (*status != CL_SUCCESS)
//    {
//        return false;
//    }

//    *status = clWaitForEvents(1, &fill_event);
    if (*status != CL_SUCCESS)
    {
        return false;
    }

    return true;
}


template<typename VALUE_TYPE, typename INDEX_TYPE>
bool executeCSRMultiply(const INDEX_TYPE m, const INDEX_TYPE n, const INDEX_TYPE nnz,
                        cl_mem alpha,
                        cl_mem row_offsets, cl_mem col_indices, cl_mem values,
                        cl_mem x,
                        cl_mem beta,
                        cl_mem y,
                        clsparseControl control,
                        const int number_of_tries
                        )
{
    const std::type_info& type_desc = typeid(VALUE_TYPE);

    if (typeid(VALUE_TYPE) != typeid(cl_float)
            && typeid(VALUE_TYPE) != typeid(cl_double))
    {
        std::cout << "[execCsrmv] Not implemented for this type "
                  << type_desc.name() <<std::endl;
        return false;
    }

    for (int i = 0; i < number_of_tries; i++)
    {
        cl_event event;
        //clsparseSetupEvent(control, &event);

        clsparseStatus spmv_status;

        if (typeid(VALUE_TYPE) == typeid(cl_float))
        {

            spmv_status = clsparseScsrmv(m, n, nnz,
                                         alpha,
                                         row_offsets, col_indices, values,
                                         x,
                                         beta,
                                         y,
                                         control);
        }
        else if(typeid(VALUE_TYPE) == typeid(cl_double))
        {
            spmv_status = clsparseDcsrmv(m, n, nnz,
                                         alpha,
                                         row_offsets, col_indices, values,
                                         x,
                                         beta,
                                         y,
                                         control);

        }
        else
        {
            std::cout << "Unknown error" << std::endl;
            return false;
        }

        clsparseStatus event_status = clsparseSuccess;
       // event_status = clsparseSynchronize(control);

        if (spmv_status != clsparseSuccess ||
                event_status != clsparseSuccess)
        {
            std::cout << "status: " << spmv_status
                      << " event: " << event_status << std::endl;

            return false;
        }
    }

    return true;
}


#endif //__MATRIX_UTILS_H__
