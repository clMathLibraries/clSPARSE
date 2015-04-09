#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include <clBLAS.h>

clsparseStatus
csr2dense_transform(const int m, const int n,
                    cl_mem row_offsets, cl_mem col_indices, cl_mem values,
                    cl_mem A,
                    const std::string& params,
                    const cl_uint group_size,
                    const cl_uint subwave_size,
                    clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,"csr2dense", "csr2dense", params);

    KernelWrap kWrapper(kernel);

    kWrapper << m << n
             << row_offsets << col_indices << values
             << A;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

    //cl::NDRange local(group_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);

    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
 
}

clsparseStatus
clsparseScsr2dense(const cl_int m, const cl_int n, const cl_int nnz,
                  cl_mem row_indices, cl_mem col_indices, cl_mem values,
                  cl_mem A,
                  clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseStatus status; 

    //validate cl_mem objects
    status = validateMemObject(A, sizeof(cl_float)*n*m);
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    //TODO: ask about validateMemObjectSize
    cl_uint nnz_per_row = nnz / m; //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = 256; //wave_size * 8;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }


    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    //fill the buffer A with zeros
    cl_float pattern = 0.0f; 
    clEnqueueFillBuffer(control->queue(), A, &pattern, sizeof(cl_float), 0, 
                        sizeof(cl_float) * m * n, NULL, NULL, NULL);

    return csr2dense_transform(m, n, 
                               row_indices, col_indices, values,
                               A,
                               params,
                               group_size,
                               subwave_size,
                               control);

}

clsparseStatus
clsparseDcsr2dense(const cl_int m, const cl_int n, const cl_int nnz,
                  cl_mem row_indices, cl_mem col_indices, cl_mem values,
                  cl_mem A,
                  clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseStatus status;

    //validate cl_mem objects
    status = validateMemObject(A, sizeof(cl_double)*n*m);
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    //TODO: ask about validateMemObjectSize

    cl_uint nnz_per_row = nnz / m; //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = wave_size * 4;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    //fill the buffer A with zeros
    cl_double pattern = 0.0f;
    clEnqueueFillBuffer(control->queue(), A, &pattern, sizeof(cl_double), 0,
                        sizeof(cl_double) * m * n, NULL, NULL, NULL);

    return csr2dense_transform(m, n,
                               row_indices, col_indices, values,
                               A,
                               params,
                               group_size,
                               subwave_size,
                               control);

}

