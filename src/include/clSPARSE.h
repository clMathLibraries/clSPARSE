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
#include "clSPARSE-1x.h"
#else
#include "clSPARSE-2x.h"
#endif

    typedef enum clsparseStatus_ {
        clsparseSuccess = CL_SUCCESS,
        clsparseInvalidValue = CL_INVALID_VALUE,
        clsparseInvalidCommandQueue = CL_INVALID_COMMAND_QUEUE,
        clsparseInvalidContext = CL_INVALID_CONTEXT,
        clsparseInvalidMemObject = CL_INVALID_MEM_OBJECT,
        clsparseInvalidDevice = CL_INVALID_DEVICE,
        clsparseInvalidEventWaitList = CL_INVALID_EVENT_WAIT_LIST,
        clsparseInvalidEvent = CL_INVALID_EVENT,
        clsparseOutOfResources = CL_OUT_OF_RESOURCES,
        clsparseOutOfHostMemory = CL_OUT_OF_HOST_MEMORY,
        clsparseInvalidOperation = CL_INVALID_OPERATION,
        clsparseCompilerNotAvailable = CL_COMPILER_NOT_AVAILABLE,
        clsparseBuildProgramFailure = CL_BUILD_PROGRAM_FAILURE,
        clsparseInvalidKernelArgs = CL_INVALID_KERNEL_ARGS,

        /* Extended error codes */
        clsparseNotImplemented = -1024, /**< Functionality is not implemented */
        clsparseNotInitialized,                 /**< clsparse library is not initialized yet */
        clsparseStructInvalid,                 /**< clsparse library is not initialized yet */
        clsparseInvalidSize,                    /**< Invalid size of object > */
        clsparseInvalidMemObj,                  /**< Checked obejct is no a valid cl_mem object */
        clsparseInsufficientMemory,             /**< The memory object for vector is too small */
        clsparseInvalidControlObject,           /**< clsparseControl object is not valid */
        clsparseInvalidFile,                    /**< Error reading the sparse matrix file */
        clsparseInvalidFileFormat,              /**< Only specific documented sparse matrix files supported */
        clsparseInvalidKernelExecution,          /**< Problem with kenrel execution */
        clsparseInvalidType,                     /** < Wrong type provided > */

        /* Solver control */
        clsparseInvalidSolverControlObject = -2048,
        clsparseInvalidSystemSize,
        clsparseIterationsExceeded,
        clsparseToleranceNotReached,
        clsparseSolverError,
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
        clsparseCreateControl( cl_command_queue queue, clsparseStatus *status );

    //enable/disable asynchronous behavior for clSPARSE;
    CLSPARSE_EXPORT clsparseStatus
        clsparseEnableAsync( clsparseControl control, cl_bool async );


    //setup events to sync
    //TODO:: NOT WORKING! NDRange throws Failure
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetupEventWaitList( clsparseControl control,
        cl_uint num_events_in_wait_list,
        cl_event* event_wait_list );

    //get the event from the last kernel execution
    CLSPARSE_EXPORT clsparseStatus
        clsparseGetEvent( clsparseControl control, cl_event* event );

    // just sets the fields to 0 or Null and free allocated struc.
    // We do not own the queue, context, etc;
    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseControl( clsparseControl control );

    /*
     * Solver control: Object controlling the solver execution
     */
    typedef enum _print_mode{
        QUIET = 0,
        NORMAL,
        VERBOSE
    } PRINT_MODE;

    typedef enum _precond
    {
        NOPRECOND = 0,
        DIAGONAL
    } PRECONDITIONER;

    typedef struct _solverControl*  clSParseSolverControl;

    CLSPARSE_EXPORT clSParseSolverControl
    clsparseCreateSolverControl(PRECONDITIONER precond, cl_int maxIters,
                            cl_double relTol, cl_double absTol);

    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseSolverControl( clSParseSolverControl solverControl );

    //here maybe some other solver control utils;
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetSolverParams( clSParseSolverControl solverControl,
                        PRECONDITIONER precond,
                        cl_int maxIters, cl_double relTol, cl_double absTol);

    CLSPARSE_EXPORT clsparseStatus
        clsparseSolverPrintMode( clSParseSolverControl solverControl, PRINT_MODE mode );

    /* Conjugate Gradients solver */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrcg( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
        clSParseSolverControl solverControl, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDcsrcg(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);

/* Bi-Conjugate Gradients Stabilized solver */
CLSPARSE_EXPORT clsparseStatus
clsparseScsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control);
    //  Library initialization and deconstruction functions
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetup( void );

    CLSPARSE_EXPORT clsparseStatus
        clsparseTeardown( void );

    CLSPARSE_EXPORT clsparseStatus
        clsparseGetVersion( cl_uint *major, cl_uint *minor, cl_uint *patch, cl_uint *tweak );

    // Convenience sparse matrix construction functions
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitScalar( clsparseScalar* scalar );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitVector( cldenseVector* vec );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCooMatrix( clsparseCooMatrix* cooMatx );

    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx );

    CLSPARSE_EXPORT clsparseStatus
        cldenseInitMatrix( cldenseMatrix* denseMatx );

    // Convenience functions provided by library to read sparse matrices from file
    CLSPARSE_EXPORT clsparseStatus
        clsparseHeaderfromFile( cl_int* nnz, cl_int* row, cl_int* col, const char* filePath );

    CLSPARSE_EXPORT clsparseStatus
        clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );

//    CLSPARSE_EXPORT clsparseStatus
//        clsparseCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    // Functions needed to compute SpM-dV operations with CSR-adaptive algorithms
    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseSCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaCompute( clsparseCsrMatrix* csrMatx, clsparseControl control );

    /* BLAS 1 routines for dense vector*/

    /* SCALE y = alpha * y */

    CLSPARSE_EXPORT clsparseStatus
        cldenseSscale( cldenseVector* y,
        const clsparseScalar* alpha,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDscale( cldenseVector* y,
        const clsparseScalar* alpha,
        const clsparseControl control );

    /* AXPY: y = alpha*x + y*/
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpy( cldenseVector* y,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpy( cldenseVector* y,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseControl control );

    /* AXPY: y = alpha*x + beta*y*/
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpby( cldenseVector* y,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseScalar* beta,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpby( cldenseVector* y,
        const clsparseScalar* alpha, const cldenseVector* x,
        const clsparseScalar* beta,
        const clsparseControl control );

    /* Reduce (sum) */
    CLSPARSE_EXPORT clsparseStatus
        cldenseIreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseSreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDreduce( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    /* norms */

    /* L1 norm */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm1( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm1( clsparseScalar *s,
        const cldenseVector* x,
        const clsparseControl control );

    /* L2 norm */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm2( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm2( clsparseScalar* s,
        const cldenseVector* x,
        const clsparseControl control );

    /* dot product */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSdot( clsparseScalar* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        cldenseDdot( clsparseScalar* r,
        const cldenseVector* x,
        const cldenseVector* y,
        const clsparseControl control );

    // BLAS 2 routines
    // SpM-dV
    // y = \alpha * A * x + \beta * y

    //new possible implementation of csrmv with control object
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmv( const clsparseScalar* alpha,
        const clsparseCsrMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrmv( const clsparseScalar* alpha,
        const clsparseCsrMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );


    CLSPARSE_EXPORT clsparseStatus
        clsparseScoomv( const clsparseScalar* alpha,
        const clsparseCooMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcoomv( const clsparseScalar* alpha,
        const clsparseCooMatrix* matx,
        const cldenseVector* x,
        const clsparseScalar* beta,
        cldenseVector* y,
        const clsparseControl control );

    // Sparse BLAS 3 routines
    // SpM-dM
    // C = \alpha * A * B  + \beta * C
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmm( const clsparseScalar* alpha,
        const clsparseCsrMatrix* sparseMatA,
        const cldenseMatrix* denseMatB,
        const clsparseScalar* beta,
        cldenseMatrix* denseMatC,
        const clsparseControl control );

    //CSR <--> Dense transformation routines
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsr2dense( const clsparseCsrMatrix* csr,
        cldenseMatrix* A,
        const clsparseControl control );

    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsr2dense( const clsparseCsrMatrix* csr,
        cldenseMatrix* A,
        clsparseControl control );

    //CSR <--> COO transformation routines
//    CLSPARSE_EXPORT clsparseStatus
//        clsparseScsr2coo( const cl_int m, const cl_int n, const cl_int nnz,
//        cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
//        cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
//        clsparseControl control );

//    CLSPARSE_EXPORT clsparseStatus
//       clsparseDcsr2coo( const cl_int m, const cl_int n, const cl_int nnz,
//        cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
//        cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
//        clsparseControl control );

//COO <--> CSR
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
                 const clsparseControl control);

//COO <--> CSR
CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
                 const clsparseControl control);

//TODO where should we put this for internal use
//    CLSPARSE_EXPORT clsparseStatus
//        clsparseScoo2csr_host( clsparseCsrMatrix* csrMatx, const clsparseCooMatrix* cooMatx, clsparseControl control );

CLSPARSE_EXPORT clsparseStatus
clsparseDcoomv(const clsparseScalar* alpha,
               const clsparseCooMatrix* matx,
               const cldenseVector* x,
               const clsparseScalar* beta,
               cldenseVector* y,
               const clsparseControl control);

//CSR <--> Dense transformation routines
CLSPARSE_EXPORT clsparseStatus
clsparseScsr2dense(const clsparseCsrMatrix* csr,
                   cldenseMatrix* A,
                   const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDcsr2dense(const clsparseCsrMatrix* csr,
                   cldenseMatrix* A,
                   clsparseControl control);

//CSR <--> COO transformation routines
//CLSPARSE_EXPORT clsparseStatus
//clsparseScsr2coo(const cl_int m, const cl_int n, const cl_int nnz,
//                 cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
//                 cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
//                 clsparseControl control);

//CLSPARSE_EXPORT clsparseStatus
//clsparseDcsr2coo(const cl_int m, const cl_int n, const cl_int nnz,
//                 cl_mem csr_row_indices, cl_mem csr_col_indices, cl_mem csr_values,
//                 cl_mem coo_row_indices, cl_mem coo_col_indices, cl_mem coo_values,
//                 clsparseControl control);

//COO <--> CSR
CLSPARSE_EXPORT clsparseStatus
clsparseScoo2csr(const clsparseCooMatrix* coo,
                 clsparseCsrMatrix* csr,
                 const clsparseControl control);

//COO <--> CSR
CLSPARSE_EXPORT clsparseStatus
clsparseDcoo2csr(const clsparseCooMatrix* coo,
                 clsparseCsrMatrix* csr,
                 const clsparseControl control);

//DENSE <--> CSR
CLSPARSE_EXPORT clsparseStatus
clsparseSdense2csr(clsparseCsrMatrix* csr,
                   const cldenseMatrix* A,
                   const clsparseControl control);

CLSPARSE_EXPORT clsparseStatus
clsparseDdense2csr(clsparseCsrMatrix* csr,
                   const cldenseMatrix* A,
                   const clsparseControl control);

#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
