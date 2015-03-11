#include "clSPARSE.h"
//#include "internal/clsparse_internal.h"
#include "internal/clsparse_sources.h"
#include "internal/clsparse_validate.h"
#include "internal/clsparse_control.h"

#include <string.h>
#include <stdio.h>
#include <assert.h>

clsparseStatus
clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
              clsparseControl control)
{
    if(!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    clsparseStatus status;

    //validate input buffers
    status = validateMemObject(buff, sizeof(cl_float)*size);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(alpha, sizeof(cl_float));
    if(status != clsparseSuccess)
        return status;


    //check opencl elements
    if (control->queue == NULL)
    {
        return clsparseInvalidCommandQueue;
    }

    //check event lists
    if ( (control->num_events_in_wait_list != 0)
         && (control->event_wait_list == NULL) )
    {
        return clsparseInvalidEventWaitList;
    }

//    cl_context context;
//    cl_int ctx_status = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
//                                              sizeof(context), &context, NULL);
//    if (ctx_status != CL_SUCCESS)
//    {
//        printf("Problem with obtaining context.\n");
//        return clsparseInvalidContext;
//    }

    //context is already in control structure

    if (control->context == NULL)
    {
        printf("Context in control structure is null.\n");
        return clsparseInvalidContext;
    }


    char params[90];
    const char* format =
            "-Werror -cl-std=CL1.2 -DINDEX_TYPE=int -DVALUE_TYPE=float -DSIZE_TYPE=int -DWG_SIZE=%u";

    sprintf(params, format, 256);

#ifndef NDEBUG
    printf("params %s\n", params);
#endif

    const char* program_name = "scale"; //maybe we can use key from program sources?

    char* key = NULL;

    createKey(program_name, params, &key);

    cl_kernel kernel = get_kernel(control->queue, program_name, params, key, &status);

    if (status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buff);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &size);
    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }

    size_t local[1];
    size_t global[1];
    local[0] = 256;
    global[0] = (( size / local[0] ) + 1) * local[0];

    status = clEnqueueNDRangeKernel(control->queue, kernel, 1,
                                    NULL, global, local,
                                    control->num_events_in_wait_list,
                                    control->event_wait_list,
                                    control->event);
    if(status != CL_SUCCESS)
    {
        free(key);
        return status;
    }

    free(key);
    return clsparseSuccess;

}

//clsparseStatus
//clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
//              cl_command_queue queue,
//              cl_uint num_events_in_wait_list,
//              const cl_event *event_wait_list,
//              cl_event *event)
//{
//    if(!clsparseInitialized)
//    {
//        return clsparseNotInitialized;
//    }

//    clsparseStatus status;

//    //validate input buffers

//    //check opencl elements
//    if (queue == NULL)
//    {
//        return clsparseInvalidCommandQueue;
//    }

//    //check event lists
//    if ( (num_events_in_wait_list != 0) && (event_wait_list == NULL) )
//    {
//        return clsparseInvalidEventWaitList;
//    }

//    cl_context context;
//    cl_int ctx_status = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
//                                              sizeof(context), &context, NULL);
//    if (ctx_status != CL_SUCCESS)
//    {
//        printf("Problem with obtaining context.\n");
//        return clsparseInvalidContext;
//    }

//    char params[90];
//    const char* format =
//            "-Werror -cl-std=CL1.2 -DINDEX_TYPE=int -DVALUE_TYPE=float -DSIZE_TYPE=int -DWG_SIZE=%u";

//    sprintf(params, format, 256);

//#ifndef NDEBUG
//    printf("params %s\n", params);
//#endif

//    const char* program_name = "scale"; //maybe we can use key from program sources?

//    char* key = NULL;

//    createKey(program_name, params, &key);

//    cl_kernel kernel = get_kernel(queue, program_name, params, key, &status);

//    if (status != CL_SUCCESS)
//    {
//        free(key);
//        return status;
//    }

//    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &buff);
//    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 0); return status ; }
//    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &alpha);
//    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 1); return status ; }
//    status = clSetKernelArg(kernel, 2, sizeof(cl_int), &size);
//    if(status != CL_SUCCESS) { printf("Problem with setting arg %d \n", 2); return status ; }

//    size_t local[1];
//    size_t global[1];
//    local[0] = 256;
//    global[0] = (( size / local[0] ) + 1) * local[0];

//    status = clEnqueueNDRangeKernel(queue, kernel, 1,
//                                    NULL, global, local,
//                                    num_events_in_wait_list, event_wait_list, event);
//    if(status != CL_SUCCESS)
//    {
//        free(key);
//        return status;
//    }

//    free(key);
//    return clsparseSuccess;

//}


