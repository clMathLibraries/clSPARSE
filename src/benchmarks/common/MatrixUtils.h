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
                    clsparseCsrMatrix& csrMatx,
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
    csrMatx.rowOffsets = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    rows.size()*sizeof(INDEX_TYPE),
                                    NULL, &mem_status);
    csrMatx.colIndices = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                    cols.size()*sizeof(INDEX_TYPE),
                                    NULL, &mem_status);
    csrMatx.values = clCreateBuffer(context, CL_MEM_READ_WRITE,
                               values.size()*sizeof(VALUE_TYPE),
                               NULL, &mem_status);



    // copy host matrix arrays to device;
    mem_status |= clEnqueueWriteBuffer(queue, csrMatx.rowOffsets, 1, 0,
                                       rows.size() * sizeof(INDEX_TYPE),
                                       rows.data(),
                                       0, NULL, NULL);
    mem_status |= clEnqueueWriteBuffer(queue, csrMatx.colIndices, 1, 0,
                                       cols.size() * sizeof(INDEX_TYPE),
                                       cols.data(),
                                       0, NULL, NULL);
    mem_status |= clEnqueueWriteBuffer(queue, csrMatx.values, 1, 0,
                                       values.size() * sizeof(VALUE_TYPE),
                                       values.data(),
                                       0, NULL, NULL);

    if (mem_status != CL_SUCCESS)
    {
        clReleaseMemObject(csrMatx.colIndices);
        clReleaseMemObject(csrMatx.rowOffsets);
        clReleaseMemObject(csrMatx.values);
        *status = mem_status;
        return false;
    }

    csrMatx.m = n_rows;
    csrMatx.n = n_cols;
    csrMatx.nnz = values.size();

    *status = mem_status;
    return true;
}

template<typename VALUE_TYPE>
bool allocateVector(clsparseVector& vector, size_t size, VALUE_TYPE value,
                    cl_command_queue queue,
                    cl_context context,
                    cl_int* status)
{
    //fix for CL 1.1 support (Nvidia devices)

    std::vector<VALUE_TYPE> fill_vec(size, value);
    vector.values = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   size * sizeof(VALUE_TYPE),
                                   fill_vec.data(), status);


    if (*status != CL_SUCCESS)
    {
        return false;
    }
    vector.n = fill_vec.size();

    return true;
}

template<typename VALUE_TYPE>
bool allocateScalar(clsparseScalar& scalar, size_t size, VALUE_TYPE value,
                    cl_command_queue queue,
                    cl_context context,
                    cl_int* status)
{
    //fix for CL 1.1 support (Nvidia devices)
    {
        std::vector<VALUE_TYPE> fill_vec(size, value);
        scalar.value = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                         size * sizeof(VALUE_TYPE),
                                         fill_vec.data(), status);

    }

    if (*status != CL_SUCCESS)
    {
        return false;
    }

    return true;
}


template<typename VALUE_TYPE>
bool executeCSRMultiply(clsparseScalar& alpha,
                        clsparseCsrMatrix& csrMatx,
                        clsparseVector& x,
                        clsparseScalar& beta,
                        clsparseVector& y,
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
            spmv_status = clsparseScsrmv(&alpha, &csrMatx, &x, &beta, &y, control);
        }
        else if(typeid(VALUE_TYPE) == typeid(cl_double))
        {
            spmv_status = clsparseDcsrmv(&alpha, &csrMatx, &x, &beta, &y, control);

        }
        else
        {
            std::cout << "Unknown error" << std::endl;
            return false;
        }

        clsparseStatus event_status = clsparseSuccess;

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
