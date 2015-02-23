#include "clSPARSE.h"
#include "internal/clsparse_sources.h"
#include "internal/clsparse_validate.h"

#include <clBLAS.h>

clsparseStatus
csrmv_a1b0(const int m,
           cl_mem row_offsets, cl_mem col_indices, cl_mem values,
           cl_mem x, cl_mem y,
           const char* params,
           const cl_uint group_size,
           const cl_uint subwave_size,
           cl_command_queue queue,
           cl_uint num_events_in_wait_list,
           const cl_event *event_wait_list,
           cl_event *event)
{
    const char* program_name = "csrmv_alpha1_beta0";
    char* key = NULL;

    createKey(program_name, params, &key);

    cl_int status;
    // ASSUME kernel name == program name
    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

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
    return status;
}

clsparseStatus
csrmv_b0(const int m, cl_mem alpha,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x, cl_mem y,
         const char* params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         cl_command_queue queue,
         cl_uint num_events_in_wait_list,
         const cl_event *event_wait_list,
         cl_event *event)
{
    const char* program_name = "csrmv_beta0";
    char* key = NULL;

    createKey(program_name, params, &key);
    cl_int status;
    // ASSUME kernel name == program name
    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }
    //set kernel arguments;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &row_offsets);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &col_indices);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 3); return status ; }
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &values);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 4); return status ; }
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &x);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 6); return status ; }

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
    return status;

}

clsparseStatus
csrmv_a1(const int m,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x,
         cl_mem beta,
         cl_mem y,
         const char* params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         cl_command_queue queue,
         cl_uint num_events_in_wait_list,
         const cl_event *event_wait_list,
         cl_event *event)
{
    const char* program_name = "csrmv_alpha1";
    char* key = NULL;

    createKey(program_name, params, &key);
    cl_int status;
    // ASSUME kernel name == program name
    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

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
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &beta);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 6); return status ; }

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
    return status;

}

clsparseStatus
csrmv_b1(const int m,
         cl_mem alpha,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x,
         cl_mem y,
         const char* params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         cl_command_queue queue,
         cl_uint num_events_in_wait_list,
         const cl_event *event_wait_list,
         cl_event *event)
{
    const char* program_name = "csrmv_beta1";
    char* key = NULL;

    createKey(program_name, params, &key);
    cl_int status;
    // ASSUME kernel name == program name
    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }
    //set kernel arguments;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &row_offsets);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &col_indices);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 3); return status ; }
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &values);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 4); return status ; }
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &x);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 6); return status ; }

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
    return status;


}

clsparseStatus
csrmv(const int m,
      cl_mem alpha,
      cl_mem row_offsets, cl_mem col_indices, cl_mem values,
      cl_mem x,
      cl_mem beta,
      cl_mem y,
      const char* params,
      const cl_uint group_size,
      const cl_uint subwave_size,
      cl_command_queue queue,
      cl_uint num_events_in_wait_list,
      const cl_event *event_wait_list,
      cl_event *event)
{
    const char* program_name = "csrmv_general";
    char* key = NULL;

    createKey(program_name, params, &key);
    cl_int status;
    // ASSUME kernel name == program name
    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }
    //set kernel arguments;
    status = clSetKernelArg(kernel, 0, sizeof(cl_int), &m);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), &row_offsets);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }
    status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &col_indices);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 3); return status ; }
    status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &values);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 4); return status ; }
    status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &x);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 5); return status ; }
    status = clSetKernelArg(kernel, 6, sizeof(cl_mem), &beta);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 6); return status ; }
    status = clSetKernelArg(kernel, 7, sizeof(cl_mem), &y);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 7); return status ; }

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
    return status;

}


clsparseStatus
clsparseScsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
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

    //check alpha and beta to distinct kernel version
    void* h_alpha = clEnqueueMapBuffer(queue, alpha, true, CL_MAP_READ,
                                       0, sizeof(float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(queue, alpha, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_alpha = %f\n", *(float*)h_alpha);
#endif
    void* h_beta = clEnqueueMapBuffer(queue, beta, true, CL_MAP_READ,
                                      0, sizeof(float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(queue, beta, h_beta, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_beta = %f\n", *(float*)h_beta);
#endif

    // this functionallity can be implemented in one kernel by using ifdefs
    // passed in parmeters but this way i found more clear;
    if(*(float*)h_alpha == 1.0f && *(float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1, beta = 0\n\n");
#endif
        //y = A*x
        return csrmv_a1b0(m, row_offsets, col_indices, values,
                          x, y,
                          params,
                          group_size,
                          subwave_size,
                          queue,
                          num_events_in_wait_list, event_wait_list, event);
    }
    else if( *(float*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, beta =/= 0 (clBlasSscale)\n\n");
#endif
        // y = b*y;
        return clblasSscal(m, *(cl_float*)h_beta, y, 0, 1, 1, &queue,
                           num_events_in_wait_list, event_wait_list, event);

    }

    else if( *(float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\nalpha =/= 0, beta = 0\n\n");
#endif
        //y = alpha * A * x
        return csrmv_b0(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x, y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }

    else if ( *(float*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(m,
                        row_offsets, col_indices, values,
                        x,
                        beta,
                        y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }

    else if ( *(float*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x,
                        y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }

    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(m,
                     alpha,
                     row_offsets,
                     col_indices,
                     values,
                     x,
                     beta,
                     y,
                     params, group_size, subwave_size, queue,
                     num_events_in_wait_list, event_wait_list, event);
    }

    return clsparseNotImplemented;
}


clsparseStatus
clsparseDcsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
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

    //check alpha and beta to distinct kernel version
    void* h_alpha = clEnqueueMapBuffer(queue, alpha, true, CL_MAP_READ,
                                       0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(queue, alpha, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("halpha = %g\n", *(cl_double*)h_alpha);
#endif
    void* h_beta = clEnqueueMapBuffer(queue, beta, true, CL_MAP_READ,
                                      0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(queue, beta, h_beta, 0, NULL, NULL);

#ifndef NDEBUG
    printf("hbeta= %g\n", *(cl_double*)h_beta);
#endif

    // this functionallity can be implemented in one kernel by using ifdefs
    // passed in parmeters but this way i found more clear;
    if(*(cl_double*)h_alpha == 1.0 && *(cl_double*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1, beta = 0\n\n");
#endif
        //y = A*x
        return csrmv_a1b0(m, row_offsets, col_indices, values,
                          x, y,
                          params,
                          group_size,
                          subwave_size,
                          queue,
                          num_events_in_wait_list, event_wait_list, event);
    }
    else if( *(cl_double*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, beta =/= 0 (clBlasDscale)\n\n");
#endif
        return clblasDscal(m, *(cl_double*)h_beta, y, 0, 1, 1, &queue,
                           num_events_in_wait_list, event_wait_list, event);
    }

    else if( *(cl_double*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 0\n\n");
#endif
        //y = alpha * A * x;
        return csrmv_b0(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x, y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }

    else if ( *(cl_double*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(m,
                        row_offsets, col_indices, values,
                        x,
                        beta,
                        y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }

    else if ( *(cl_double*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x,
                        y,
                        params, group_size, subwave_size, queue,
                        num_events_in_wait_list, event_wait_list, event);
    }
    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(m,
                     alpha,
                     row_offsets,
                     col_indices,
                     values,
                     x,
                     beta,
                     y,
                     params, group_size, subwave_size, queue,
                     num_events_in_wait_list, event_wait_list, event);
    }

    return clsparseNotImplemented;
}
