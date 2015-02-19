#include "clSPARSE.h"
#include "internal/clsparse_sources.h"
#include "internal/clsparse_validate.h"

clsparseStatus
clsparseScsrmv(const int m, const int n, const int nnz,
               const cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_mem beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    clsparseStatus status;

    status = validateMemObject(x, sizeof(cl_float)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_float)*m);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(alpha, sizeof(cl_float));
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(beta, sizeof(cl_float));
    if(status != clsparseSuccess)
        return status;

    //check queue
    if (queue == NULL)
    {
        return clsparseInvalidCommandQueue;
    }

    //check event lists
    if ( (num_events_in_wait_list != 0) && (event_wait_list == NULL) )
    {
        return clsparseInvalidEventWaitList;
    }

    cl_uint nnz_per_row = nnz / m; //average nnz per row
    cl_uint wave_size = 64;
    cl_uint group_size = wave_size * 4;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if (nnz_per_row < 64) {  subwave_size = 32;  }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    char params [128];
    const char* format =
        "-Werror -DINDEX_TYPE=uint -DVALUE_TYPE=float -DSIZE_TYPE=uint -DWG_SIZE=%u -DSUBWAVE_SIZE=%u";

    //inprint the kernel parameters into the params;
    sprintf(params, format, group_size, subwave_size);

#ifndef NDEBUG
    printf("params %s\n", params);
#endif

    const char* program_name = "csrmv_vector";
    char* key = NULL;

    createKey(program_name, params, &key);

    //ASSUME kernel name == program name
    cl_kernel kernel = getKernel(queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    //set kernel arguments;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &row_offsets);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &col_indices);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &values);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 3); return status ; }
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &x);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 4); return status ; }
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }

    // predicted number of subwaves to be executed;
    // subwave takes care of each row in matrix;
    cl_uint predicted = subwave_size * m;

    size_t local[1];
    size_t global[1];
    local[0] = group_size;
    global[0] = predicted > local[0] ? predicted : local[0];

    status = clEnqueueNDRangeKernel(queue, kernel, 1,
                                    NULL, global, local,
                                    num_events_in_wait_list, event_wait_list, event);
    if(status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    free(key);
    return clsparseSuccess;
}


clsparseStatus
clsparseDcsrmv(const int m, const int n, const int nnz,
               const cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_mem beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    clsparseStatus status;

    status = validateMemObject(x, sizeof(cl_double)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_double)*m);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(alpha, sizeof(cl_double));
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(beta, sizeof(cl_double));
    if(status != clsparseSuccess)
        return status;

    //check queue
    if (queue == NULL)
    {
        return clsparseInvalidCommandQueue;
    }

    //check event lists
    if ( (num_events_in_wait_list != 0) && (event_wait_list == NULL) )
    {
        return clsparseInvalidEventWaitList;
    }

    cl_uint nnz_per_row = nnz / m; //average nnz per row
    cl_uint wave_size = 64;
    cl_uint group_size = wave_size * 4;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if (nnz_per_row < 64) {  subwave_size = 32;  }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    char params [128];
    const char* format =
        "-Werror -DINDEX_TYPE=uint -DVALUE_TYPE=double -DSIZE_TYPE=uint -DWG_SIZE=%u -DSUBWAVE_SIZE=%u";

    //inprint the kernel parameters into the params;
    sprintf(params, format, group_size, subwave_size);

#ifndef NDEBUG
    printf("params %s\n", params);
#endif

    const char* program_name = "csrmv_vector";
    char* key = NULL;

    createKey(program_name, params, &key);

    //ASSUME kernel name == program name
    cl_kernel kernel = getKernel(queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    //set kernel arguments;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &row_offsets);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &col_indices);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &values);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 3); return status ; }
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &x);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 4); return status ; }
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }

    // predicted number of subwaves to be executed;
    // subwave takes care of each row in matrix;
    cl_uint predicted = subwave_size * m;

    size_t local[1];
    size_t global[1];
    local[0] = group_size;
    global[0] = predicted > local[0] ? predicted : local[0];

    status = clEnqueueNDRangeKernel(queue, kernel, 1,
                                    NULL, global, local,
                                    num_events_in_wait_list, event_wait_list, event);
    if(status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    free(key);
    return clsparseSuccess;
}
