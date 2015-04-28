#pragma once
#ifndef _CL_SPARSE_H_
#define _CL_SPARSE_H_

// CMake-generated file to define export related preprocessor macros
#include "clsparse_export.h"

#ifdef __cplusplus
extern "C" {
#endif

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION < 200 )
    #include "clSPARSE_1x.h"
#else
    #include "clSPARSE_2x.h"
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
    clsparseStructInvalid,                 /**< clsparse library is not initialized yet */
    clsparseInvalidSize,                    /**< Invalid size of object > */
    clsparseInvalidMemObj,                  /**< Checked obejct is no a valid cl_mem object */
    clsparseInsufficientMemory,             /**< The memory object for vector is too small */
    clsparseInvalidControlObject,           /**< clsparseControl object is not valid */
    clsparseInvalidFile,                    /**< Error reading the sparse matrix file */
    clsparseInvalidFileFormat,              /**< Only specific documented sparse matrix files supported */
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


//Deprecated: offsets are hidden in type structures.
//CLSPARSE_EXPORT clsparseStatus
//clsparseSetOffsets(clsparseControl control,
//                   size_t off_alpha, size_t off_beta,
//                   size_t off_x, size_t off_y);


CLSPARSE_EXPORT clsparseStatus
clsparseGetVersion( cl_uint *major, cl_uint *minor, cl_uint *patch, cl_uint *tweak );

CLSPARSE_EXPORT clsparseStatus
clsparseSetup(void);

CLSPARSE_EXPORT clsparseStatus
clsparseTeardown(void);

// THIS IS JUST AN TEST KERNEL FOR TESTING PURPOSES
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
clsparseInitDenseMatrix( clsparseDenseMatrix* denseMatx );

CLSPARSE_EXPORT clsparseStatus
clsparseCooHeaderfromFile( clsparseCooMatrix* cooMatx, const char* filePath );

CLSPARSE_EXPORT clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseScoo2csr( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseCsrComputeMeta( clsparseCsrMatrix* csrMatx, clsparseControl control );

// SPMV
// y = \alpha * A * x + \beta * y
// TODO:: alpha, beta should we provide them as cl_mem buffers?
// TODO:: add matrixDescriptor for particular matrix properties
//        like avg nnz per row which help to determine kernel scalar vector,
//        it is simple for csr  avg nnz per row = nnz / n_rows
//        type of matrix (GENERAL, SYMMETRIC(faster spmv) etc.
//        index based (0, 1)

//new possible implementation of csrmv with control object
CLSPARSE_EXPORT clsparseStatus
clsparseScsrmv( const clsparseScalar* alpha,
                const clsparseCsrMatrix* matx,
                const clsparseVector* x,
                const clsparseScalar* beta,
                clsparseVector* y,
                const clsparseControl control );

//What is it for?
CLSPARSE_EXPORT clsparseStatus
clsparseScsrmv_adaptive_scalar( float alpha,
            const clsparseCsrMatrix* matx,
            const clsparseVector* x,
            float beta,
            const clsparseVector* y,
            const clsparseControl control );


CLSPARSE_EXPORT clsparseStatus
clsparseDcsrmv( const clsparseScalar* alpha,
                const clsparseCsrMatrix* matx,
                const clsparseVector* x,
                const clsparseScalar* beta,
                clsparseVector* y,
                const clsparseControl control );


CLSPARSE_EXPORT clsparseStatus
clsparseScoomv(const clsparseScalar* alpha,
               const clsparseCooMatrix* matx,
               const clsparseVector* x,
               const clsparseScalar* beta,
               clsparseVector* y,
               const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcoomv(const clsparseScalar* alpha,
               const clsparseCooMatrix* matx,
               const clsparseVector* x,
               const clsparseScalar* beta,
               clsparseVector* y,
               const clsparseControl control);

//CSR <--> Dense transformation routines
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2dense(const clsparseCsrMatrix* csr,
                   clsparseDenseMatrix* A,
                   const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2dense(const clsparseCsrMatrix* csr,
                   clsparseDenseMatrix* A,
                   clsparseControl control);


#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
