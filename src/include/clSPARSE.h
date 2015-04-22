#pragma once
#ifndef _CL_SPARSE_H_
#define _CL_SPARSE_H_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

// CMake-generated file to define export related preprocessor macros
#include "clsparse_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "clSPARSE_2x.h"
#else
    #include "clSPARSE_1x.h"
#endif

// Type definitions - to be fleshed in
//typedef enum clsparseOperation_t clsparseOperation;
//typedef enum clsparseMatDescr_t clsparseMatDescr;

typedef enum clsparseStatus_ {
    clsparseSuccess                         = CL_SUCCESS,
    clsparseInvalidValue                    = CL_INVALID_VALUE,
    clsparseInvalidCommandQueue             = CL_INVALID_COMMAND_QUEUE,
    clsparseInvalidContext                  = CL_INVALID_CONTEXT,
    clsparseInvalidMemObject                = CL_INVALID_MEM_OBJECT,
    clsparseInvalidDevice                   = CL_INVALID_DEVICE,
    clsparseInvalidEventWaitList            = CL_INVALID_EVENT_WAIT_LIST,
    clsparseInvalidEvent                    = CL_INVALID_EVENT,
    clsparseOutOfResources                  = CL_OUT_OF_RESOURCES,
    clsparseOutOfHostMemory                 = CL_OUT_OF_HOST_MEMORY,
    clsparseInvalidOperation                = CL_INVALID_OPERATION,
    clsparseCompilerNotAvailable            = CL_COMPILER_NOT_AVAILABLE,
    clsparseBuildProgramFailure             = CL_BUILD_PROGRAM_FAILURE,
    clsparseInvalidKernelArgs                = CL_INVALID_KERNEL_ARGS,

    /* Extended error codes */
    clsparseNotImplemented         = -1024, /**< Functionality is not implemented */
    clsparseNotInitialized,                 /**< clsparse library is not initialized yet */
    clsparseInvalidSize,                    /**< Invalid size of object > */
    clsparseInvalidMemObj,                  /**< Checked obejct is no a valid cl_mem object */
    clsparseInsufficientMemory,             /**< The memory object for vector is too small */
    clsparseInvalidControlObject,           /**< clsparseControl object is not valid */
    clsparseInvalidKernelExecution          /**< Problem with kenrel execution */

} clsparseStatus;


// clsparseControl keeps the data relevant for
// OpenCL operations like kernel execution, mem alocation, sync.
/* To be considered:
    - how the allocation should look like?
        IMO clsparseControl ctrl = clsparseControl { .queue = queue ... } is not nice
    - if there is sth like clsparseControl how we should destroy it? in the tearDown function?
    - if the user call the clReleaseCommandQueue the clsparseControl become invalid.
*/
//
typedef struct _clsparseControl*  clsparseControl;

//setup the control from external queue;
CLSPARSE_EXPORT clsparseControl 
clsparseCreateControl( cl_command_queue queue, cl_int *status );

//enable/disable asynchronous behavior for clSPARSE;
CLSPARSE_EXPORT clsparseStatus
clsparseEnableAsync(clsparseControl control, cl_bool async);


//setup events to sync
//TODO:: NOT WORKING! NDRange throws Failure
CLSPARSE_EXPORT clsparseStatus
clsparseSetupEventWaitList(clsparseControl control,
                           cl_uint num_events_in_wait_list,
                           cl_event* event_wait_list);

//get the event from the last kernel execution
CLSPARSE_EXPORT clsparseStatus
clsparseGetEvent(clsparseControl control, cl_event* event);

// just sets the fields to 0 or Null and free allocated struc.
// We do not own the queue, context, etc;
CLSPARSE_EXPORT clsparseStatus
clsparseReleaseControl(clsparseControl control);


CLSPARSE_EXPORT clsparseStatus
clsparseSetOffsets(clsparseControl control,
                   size_t off_alpha, size_t off_beta,
                   size_t off_x, size_t off_y);


CLSPARSE_EXPORT clsparseStatus
clsparseGetVersion( cl_uint *major, cl_uint *minor, cl_uint *patch, cl_uint *tweak );

CLSPARSE_EXPORT clsparseStatus
clsparseSetup(void);

CLSPARSE_EXPORT clsparseStatus
clsparseTeardown(void);

// THIS IS JUST AN TEST KERNEL FOR TESTING PURPOSES

//OLD Implementation;
//clsparseStatus
//clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
//              cl_command_queue queue,
//              cl_uint num_events_in_wait_list,
//              const cl_event *event_wait_list,
//              cl_event *event);

//New implementation with clsparseControl structure
CLSPARSE_EXPORT clsparseStatus
clsparseScale(cl_mem buff, cl_mem alpha, cl_int size,
              clsparseControl control);

// Convenience sparse matrix construction functions
CLSPARSE_EXPORT clsparseStatus
clsparseInitScalar( clsparseScalar* scalar );

CLSPARSE_EXPORT clsparseStatus
clsparseInitVector( clsparseVector* vec );

CLSPARSE_EXPORT clsparseStatus
clsparseInitCooMatrix( clsparseCooMatrix* cooMatx );

CLSPARSE_EXPORT clsparseStatus
clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx );

CLSPARSE_EXPORT clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath );

CLSPARSE_EXPORT clsparseStatus
clsparseCoo2Csr( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx );

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
CLSPARSE_EXPORT clsparseStatus
clsparseScsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
               clsparseControl control);

//new possible implementation of csrmv with control object
CLSPARSE_EXPORT clsparseStatus
clsparseScsrmv_adaptive( const clsparseScalar* alpha,
                const clsparseCsrMatrix* matx,
                const clsparseVector* x,
                const clsparseScalar* beta,
                clsparseVector* y,
                const clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseScsrmv_adaptive_scalar( float alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            float beta,
            const clsparseVector* y,
            clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDcsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
               clsparseControl control);


CLSPARSE_EXPORT clsparseStatus
clsparseScoomv(const cl_int m, const cl_int n, const cl_int nnz,
               cl_mem alpha,
               cl_mem row_indices, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

CLSPARSE_EXPORT clsparseStatus
clsparseDcoomv(const cl_int m, const cl_int n, const cl_int nnz,
               cl_mem alpha,
               cl_mem row_indices, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
               cl_command_queue queue,
               cl_uint num_events_in_wait_list,
               const cl_event *event_wait_list,
               cl_event *event);

//CSR <--> Dense transformation routines
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2dense(const cl_int m, const cl_int n, const cl_int nnz,
                   cl_mem row_indices, cl_mem col_indices, cl_mem values,
                   cl_mem A,
                   clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2dense(const cl_int m, const cl_int n, const cl_int nnz,
                   cl_mem row_indices, cl_mem col_indices, cl_mem values,
                   cl_mem A,
                   clsparseControl control);

//CSR <--> COO transformation routines
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2coo(const cl_int m, const cl_int n, const cl_int nnz,
                 cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
                 cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
                 clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2coo(const cl_int m, const cl_int n, const cl_int nnz,
                 cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
                 cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
                 clsparseControl control);



#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
