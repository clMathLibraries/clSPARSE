#pragma once
#ifndef _CL_SPARSE_H_
#define _CL_SPARSE_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*! This preprocessor definition is the standard way of making exporting APIs
*  from a DLL simpler. All files within this DLL are compiled with the CLSPARSE_EXPORTS
*  symbol defined on the command line. This symbol should not be defined on any project
*  that uses this DLL. This way any other project whose source files include this file see
*  clfft functions as being imported from a DLL, whereas this DLL sees symbols
*  defined with this macro as being exported.
*/
#if defined( _WIN32 )
#if !defined( __cplusplus )
    #define inline __inline
#endif

#if defined( CLSPARSE_EXPORTS )
    #define CLSPARSEAPI __declspec( dllexport )
#else
    #define CLSPARSEAPI __declspec( dllimport )
#endif
#else
    #define CLSPARSEAPI
#endif

// Type definitions - to be fleshed in
//typedef enum clsparseOperation_t clsparseOperation;
//typedef enum clsparseMatDescr_t clsparseMatDescr;


#ifdef __cplusplus
extern "C" {
#endif

typedef enum clsparseStatus_ {
    clsparseSuccess                         = CL_SUCCESS,
    clsparseInvalidValue                    = CL_INVALID_VALUE,
    clsparseInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clsparseInvalidContext                  = CL_INVALID_CONTEXT,
    clsparseInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clsparseInvalidDevice                   = CL_INVALID_DEVICE,
    clsparseInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clsparseOutOfResources                  = CL_OUT_OF_RESOURCES,
    clsparseOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clsparseInvalidOperation                = CL_INVALID_OPERATION,
    clsparseCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clsparseBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,

    /* Extended error codes */
    clsparseNotImplemented         = -1024, /**< Functionality is not implemented */
    clsparseNotInitialized,                 /**< clsparse library is not initialized yet */
    clsparseInvalidSize,                    /**< Invalid size of object > */
    clsparseInvalidMemObj                   /**< Checked obejct is no a valid cl_mem object */
} clsparseStatus;




clsparseStatus
clsparseGetVersion (cl_uint* major, cl_uint* minor, cl_uint* patch);

clsparseStatus
clsparseSetup(void);

clsparseStatus
clsparseTeardown(void);

// THIS IS JUST AN TEST KERNEL FOR TESTING PURPOSES
clsparseStatus
clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
              cl_command_queue queue,
              cl_uint num_events_in_wait_list,
              const cl_event *event_wait_list,
              cl_event *event);

// SPMV
// y = \alpha * A * x + \beta * y
// TODO:: alpha, beta scaling is not supported yet
// TODO:: alpha, beta should be cl_mem
// TODO:: add offsets for alpha and beta cl_mems
// TODO:: here only one queue is supported. Do we need more?
// TODO:: add matrixDescriptor for particular matrix properties
//        like avg nnz per row which help to determine kernel scalar vector,
//        it is simple for csr  avg nnz per row = nnz / n_rows
//        type of matrix (GENERAL, SYMMETRIC(faster spmv) etc.
//        index based (0, 1)
clsparseStatus
clsparseScsrmv(const int m, const int n, const int nnz,
               const cl_float alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_float beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

clsparseStatus
clsparseDcsrmv(const int m, const int n, const int nnz,
               const cl_double alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_double beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);


clsparseStatus
clsparseScoomv(const cl_int m, const cl_int n, const cl_int nnz,
               const cl_float alpha,
               cl_mem row_indices, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_float beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

clsparseStatus
clsparseDcoomv(const cl_int m, const cl_int n, const cl_int nnz,
               const cl_double alpha,
               cl_mem row_indices, cl_mem col_indices, cl_mem values,
               cl_mem x,
               const cl_double beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
