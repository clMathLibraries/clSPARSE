/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

 /*! \file
 * \brief clSPARSE.h defines 'C' compatible callable functions and types that
 * call into the library
 */

#pragma once
#ifndef _CL_SPARSE_H_
#define _CL_SPARSE_H_

// CMake-generated file to define export related preprocessor macros
#include "clsparse_export.h"

#ifdef __cplusplus
extern "C" {
#endif

/*! \brief Building for OpenCL version is a compile time decision
*/
#if( BUILD_CLVERSION < 200 )
#include "clSPARSE-1x.h"
#else
#include "clSPARSE-2x.h"
#endif

    /*! \brief Possible error code values that can be returned from clSPARSE API's
    */
    typedef enum clsparseStatus_
    {
        /** @name Inherited OpenCL codes */
        /**@{*/
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
        /**@}*/

        /** @name Extended error codes */
        /**@{*/
        clsparseNotImplemented = -1024,  /**< Functionality is not implemented */
        clsparseNotInitialized,          /**< clsparse library is not initialized yet */
        clsparseStructInvalid,           /**< clsparse library is not initialized yet */
        clsparseInvalidSize,             /**< Invalid size of object > */
        clsparseInvalidMemObj,           /**< Checked object is no a valid cl_mem object */
        clsparseInsufficientMemory,      /**< The memory object for vector is too small */
        clsparseInvalidControlObject,    /**< clsparseControl object is not valid */
        clsparseInvalidFile,             /**< Error reading the sparse matrix file */
        clsparseInvalidFileFormat,       /**< Only specific documented sparse matrix files supported */
        clsparseInvalidKernelExecution,  /**< Problem with kernel execution */
        clsparseInvalidType,             /** < Wrong type provided > */
        /**@}*/

        /** @name Solver control codes */
        /**@{*/
        clsparseInvalidSolverControlObject = -2048,
        clsparseInvalidSystemSize,
        clsparseIterationsExceeded,
        clsparseToleranceNotReached,
        clsparseSolverError,
        /**@}*/
    } clsparseStatus;


    /*!
    * \defgroup SETUP Library setup or teardown functions
    *
    * \brief Functions to control the lifetime of the clSPARSE library
    * \example sample-initialization-c.c
    * \example sample-initialization.cpp
    */
    /**@{*/

    /*!
    * \brief Initialize the clsparse library
    * \note Must be called before any other clsparse API function is invoked.
    *
    * \returns clsparseSuccess
    *
    * \ingroup SETUP
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetup( void );

    /*!
    * \brief Finalize the usage of the clsparse library
    * Frees all state allocated by the clsparse runtime and other internal data
    *
    * \returns clsparseSuccess
    *
    * \ingroup SETUP
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseTeardown( void );

    /*!
    * \brief Query for the runtime clsparse library version info
    *
    * \param[out] major   library's major version
    * \param[out] minor   library's minor version
    * \param[out] patch   library's patch version
    * \param[out] tweak   library's tweak version
    *
    * \returns clsparseSuccess
    *
    * \ingroup SETUP
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseGetVersion( cl_uint* major, cl_uint* minor, cl_uint* patch, cl_uint* tweak );
    /**@}*/

    /*!
    * \defgroup INIT Routines to initialize a clsparse object
    *
    * \brief Initialize a clsparse data structure to default values
    */
    /**@{*/

    /*!
    * \brief Initialize a scalar structure to be used in the clsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] scalar  Scalar structure to be initialized
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup INIT
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitScalar( clsparseScalar* scalar );

    /*!
    * \brief Initialize a dense vector structure to be used in the clsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] vec  Dense vector structure to be initialized
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup INIT
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitVector( cldenseVector* vec );

    /*!
    * \brief Initialize a sparse matrix COO structure to be used in the clsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] cooMatx  Sparse COO matrix structure to be initialized
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup INIT
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCooMatrix( clsparseCooMatrix* cooMatx );

    /*!
    * \brief Initialize a sparse matrix CSR structure to be used in the clsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] csrMatx  Sparse CSR matrix structure to be initialized
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup INIT
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseInitCsrMatrix( clsparseCsrMatrix* csrMatx );

    /*!
    * \brief Initialize a dense matrix structure to be used in the clsparse library
    * \note It is users responsibility to allocate OpenCL device memory
    *
    * \param[out] denseMatx  Dense matrix structure to be initialized
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup INIT
    */
    CLSPARSE_EXPORT clsparseStatus
        cldenseInitMatrix( cldenseMatrix* denseMatx );
    /**@}*/

    /*!
    * \defgroup STATE Modifying library state
    *
    * \brief Functions to create or modify library state
    */
    /**@{*/

    /*! \brief clsparseControl keeps state relevant for OpenCL operations
     * like kernel execution, memory allocation and synchronization behavior
     */
    typedef struct _clsparseControl*  clsparseControl;

    /*!
    * \brief setup the clsparse control object from external OpenCL queue
    *
    * \param[in] queue   cl_command_queue created
    * \param[out] status   clsparse error return value from function
    *
    * \returns \b On successful completion, a valid clsparseControl object
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseControl
        clsparseCreateControl( cl_command_queue queue, clsparseStatus *status );

    /*!
    * \brief Enable/Disable asynchronous behavior for clSPARSE
    *
    * \param[in] control  A valid clsparseControl created with clsparseCreateControl
    * \param[in] async  True to enable immediate return, false to block execution until event completion
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseEnableAsync( clsparseControl control, cl_bool async );

    /*!
    * \brief Enable/Disable the use of compensated summation
    * \note This currently only controls the precision of SpM-dV
    *
    * \param[in] control   A valid clsparseControl created with clsparseCreateControl
    * \param[in] extPrecision   True to enable extended precision, false for fast precision
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseEnableExtendedPrecision( clsparseControl control, cl_bool extPrecision );

    /*!
    * \brief Configure the library to use an array of events
    * \warning NOT WORKING! NDRange throws Failure
    *
    * \param[in] control  A valid clsparseControl created with clsparseCreateControl
    * \param[in] num_events_in_wait_list   Size of the event_wait_list array
    * \param[out] event_wait_list   An array of OpenCL event objects for client to wait on
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetupEventWaitList( clsparseControl control,
                                    cl_uint num_events_in_wait_list,
                                    cl_event* event_wait_list );

    /*!
    * \brief Return an event from the last kernel execution
    *
    * \param[in] control  A valid clsparseControl created with clsparseCreateControl
    * \param[out] event  The returned event for the last kernel queued into the cl_command_queue inside the
    * clsparseControl object
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseGetEvent( clsparseControl control, cl_event* event );

    /*!
    * \brief Sets internal control fields to 0 or Null and frees allocated structures
    *
    * \param[in,out] control  A valid clsparseControl created with clsparseCreateControl
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup STATE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseControl( clsparseControl control );
    /**@}*/

    /*!
     * \defgroup SOLVER Sparse iterative solvers
     *
     * \brief Functions to setup or execute sparse iterative solvers
     * \example sample-cg.cpp
     */
     /**@{*/

    /*! \brief Enumeration to control the verbosity of the sparse iterative
     * solver routines.  VERBOSE will print helpful diagnostic messages to
     * console
     *
     * \ingroup SOLVER
     */
    typedef enum _print_mode
    {
        QUIET = 0,
        NORMAL,
        VERBOSE
    } PRINT_MODE;

    /*! \brief Enumeration to select the pre-conditioner algorithm used pre-conditioner
     * the sparse data before the iterative solvers execution
     *
     * \ingroup SOLVER
     */
    typedef enum _precond
    {
        NOPRECOND = 0,
        DIAGONAL
    } PRECONDITIONER;

    /*! \brief clSParseSolverControl keeps state relevant for OpenCL operations
     * like kernel execution, memory allocation and synchronization behavior,
     * specifically for sparse iterative solvers
     *
     * \ingroup SOLVER
     */
    typedef struct _solverControl*  clSParseSolverControl;

    /*!
    * \brief Create a clSParseSolverControl object to control clsparse iterative
    * solver operations
    *
    * \param[in] precond  A valid enumeration constant from PRECONDITIONER
    * \param[in] maxIters  Maximum number of iterations to converge before timing out
    * \param[in] relTol  Relative tolerance
    * \param[in] absTol  Absolute tolerance
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup SOLVER
    */
    CLSPARSE_EXPORT clSParseSolverControl
        clsparseCreateSolverControl( PRECONDITIONER precond, cl_int maxIters,
                                     cl_double relTol, cl_double absTol );

     /*!
     * \brief Release a clSParseSolverControl object created with clsparseCreateSolverControl
     *
     * \param[in,out] solverControl  clSPARSE object created with clsparseCreateSolverControl
     *
     * \returns \b clsparseSuccess
     *
     * \ingroup SOLVER
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseReleaseSolverControl( clSParseSolverControl solverControl );

    /*!
    * \brief Set clSParseSolverControl state
    *
    * \param[in] solverControl  clSPARSE object created with clsparseCreateSolverControl
    * \param[in] precond A valid enumeration constant from PRECONDITIONER, how to precondition sparse data
    * \param[in] maxIters  Maximum number of iterations to converge before timing out
    * \param[in] relTol  Relative tolerance
    * \param[in] absTol  Absolute tolerance
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup SOLVER
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSetSolverParams( clSParseSolverControl solverControl,
                                 PRECONDITIONER precond,
                                 cl_int maxIters, cl_double relTol, cl_double absTol );

     /*!
     * \brief Set the verbosity level of the clSParseSolverControl object
     *
     * \param[in] solverControl  clSPARSE object created with clsparseCreateSolverControl
     * \param[in] mode A valid enumeration constant from PRINT_MODE, to specify verbosity level
     *
     * \returns \b clsparseSuccess
     *
     * \ingroup SOLVER
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSolverPrintMode( clSParseSolverControl solverControl, PRINT_MODE mode );

    /*!
    * \brief Execute a single precision Conjugate Gradients solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a clSPARSE CSR matrix with single precision data
    * \param[in] b  the input dense vector with single precision data
    * \param[in] solverControl  a valid clSParseSolverControl object created with clsparseCreateSolverControl
    * \param[in] control A valid clsparseControl created with clsparseCreateControl
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup SOLVER
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrcg( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
                        clSParseSolverControl solverControl, clsparseControl control );

    /*!
    * \brief Execute a double precision Conjugate Gradients solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a clSPARSE CSR matrix with double precision data
    * \param[in] b  the input dense vector with double precision data
    * \param[in] solverControl  a valid clSParseSolverControl object created with clsparseCreateSolverControl
    * \param[in] control A valid clsparseControl created with clsparseCreateControl
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup SOLVER
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrcg( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
                        clSParseSolverControl solverControl, clsparseControl control );

     /*!
     * \brief Execute a single precision Bi-Conjugate Gradients Stabilized solver
     *
     * \param[in] x  the dense vector to solve for
     * \param[in] A  the clSPARSE CSR matrix with single precision data
     * \param[in] b  the input dense vector with single precision data
     * \param[in] solverControl  a valid clSParseSolverControl object created with clsparseCreateSolverControl
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \returns \b clsparseSuccess
     *
     * \ingroup SOLVER
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrbicgStab( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
                              clSParseSolverControl solverControl, clsparseControl control );

    /*!
    * \brief Execute a double precision Bi-Conjugate Gradients Stabilized solver
    *
    * \param[in] x  the dense vector to solve for
    * \param[in] A  a clSPARSE CSR matrix with double precision data
    * \param[in] b  the input dense vector with double precision data
    * \param[in] solverControl  a valid clSParseSolverControl object created with clsparseCreateSolverControl
    * \param[in] control A valid clsparseControl created with clsparseCreateControl
    *
    * \returns \b clsparseSuccess
    *
    * \ingroup SOLVER
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrbicgStab( cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
                              clSParseSolverControl solverControl, clsparseControl control );
    /**@}*/

    /*!
    * \defgroup FILE Support functions provided to read sparse matrices from file
    *
    * \brief Functions to help read the contents of matrix market files from disk
    */
    /**@{*/

    /*!
    * \brief Read the sparse matrix header from file
    *
    * \param[out] nnz  The number of non-zeroes present in the sparse matrix structure
    * \param[out] row  The number of rows in the sparse matrix
    * \param[out] col  The number of columns in the sparse matrix
    * \param[in] filePath  A path in the file-system to the sparse matrix file
    *
    * \note At this time, only matrix market (.MTX) files are supported
    * \warning The value returned in nnz is the maximum possible number of non-zeroes from the sparse
    * matrix on disk (can be used to allocate memory).  The actual number of non-zeroes may be less,
    * depending if explicit zeroes were stored in file.
    * \returns \b clsparseSuccess
    *
    * \ingroup FILE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseHeaderfromFile( cl_int* nnz, cl_int* row, cl_int* col, const char* filePath );

    /*!
    * \brief Read sparse matrix data from file in single precision COO format
    * \details This function reads the contents of the sparse matrix file into clsparseCooMatrix data structure.
    * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
    * This function sorts the values read (on host) by row, then column before copying them into
    * device memory
    * \param[out] cooMatx  The COO sparse structure that represents the matrix in device memory
    * \param[in] filePath  A path in the file-system to the sparse matrix file
    * \param[in] control A valid clsparseControl created with clsparseCreateControl
    *
    * \note The number of non-zeroes actually read from the file may be less than the number of
    * non-zeroes reported from the file header
    * \note The OpenCL device memory must be allocated before the call to this function.
    * \returns \b clsparseSuccess
    *
    * \ingroup FILE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );

    /*!
     * \brief Read sparse matrix data from file in double precision COO format
     * \details This function reads the contents of the sparse matrix file into clsparseCooMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory.  If the data on disk is stored in single precision, this function will
     * up-convert the values to double.
     * \param[out] cooMatx  The COO sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \returns \b clsparseSuccess
     *
     * \ingroup FILE
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control );

    /*!
     * \brief Read sparse matrix data from file in single precision CSR format
     * \details This function reads the contents of the sparse matrix file into clsparseCsrMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory
     * \param[out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \returns \b clsparseSuccess
     *
     * \ingroup FILE
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    /*!
     * \brief Read sparse matrix data from file in double precision CSR format
     * \details This function reads the contents of the sparse matrix file into clsparseCsrMatrix data structure.
     * The data structure represents the contents of the sparse matrix data in OpenCL device memory.
     * This function sorts the values read (on host) by row, then column before copying them into
     * device memory.  If the data on disk is stored in single precision, this function will
     * up-convert the values to double.
     * \param[out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] filePath  A path in the file-system to the sparse matrix file
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \note The number of non-zeroes actually read from the file may be less than the number of
     * non-zeroes reported from the file header
     * \note The OpenCL device memory must be allocated before the call to this function.
     * \returns \b clsparseSuccess
     *
     * \ingroup FILE
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDCsrMatrixfromFile( clsparseCsrMatrix* csrMatx, const char* filePath, clsparseControl control );

    /*!
     * \brief Calculate the amount of device memory required to hold meta-data for csr-adaptive SpM-dV algorithm
     * \details CSR-adaptive is a high performance sparse matrix times dense vector algorithm.  It requires a pre-processing
     * step to calculate meta-data on the sparse matrix.  This meta-data is stored alongside and carried along
     * with the other matrix data.  This function initializes the rowBlockSize member variable of the csrMatx
     * variable with the appropriate size.  The client program is responsible to allocate device memory in rowBlocks
     * of this size before calling into the library compute routines.
     * \param[in,out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup FILE
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaSize( clsparseCsrMatrix* csrMatx, clsparseControl control );

    /*!
     * \brief Calculate the meta-data for csr-adaptive SpM-dV algorithm
     * \details CSR-adaptive is a high performance sparse matrix times dense vector algorithm.  It requires a pre-processing
     * step to calculate meta-data on the sparse matrix.  This meta-data is stored alongside and carried along
     * with the other matrix data.  This function calculates the meta data and stores it into the rowBlocks member of
     * the clsparseCsrMatrix.
     * \param[in,out] csrMatx  The CSR sparse structure that represents the matrix in device memory
     * \param[in] control  A valid clsparseControl created with clsparseCreateControl
     * \note This function assumes that the memory for rowBlocks has already been allocated by client program
     *
     * \ingroup FILE
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseCsrMetaCompute( clsparseCsrMatrix* csrMatx, clsparseControl control );
    /**@}*/

    /*!
     * \defgroup BLAS clSPARSE BLAS operations
     *
     * \brief BLAS linear algebra
     * \details BLAS is categorized into 3 groups: L1, L2 & L3
     */

    /*!
     * \defgroup BLAS-1 Dense L1 BLAS operations
     *
     * \brief Dense BLAS level 1 routines for dense vectors
     * \details These L1 BLAS functions were developed internally to clsparse during its development
     * and made available through an API.  The thought is that these routines could be useful to
     * others as performance primitives writing their own sparse operations
     *
     * \ingroup BLAS
     * \example sample-axpy.cpp
     * \example sample-norm1-c.c
     */
    /**@{*/

    /*!
     * \brief Single precision scale dense vector by a scalar
     * \details \f$ r \leftarrow \alpha \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSscale( cldenseVector* r,
                       const clsparseScalar* alpha,
                       const cldenseVector* y,
                       const clsparseControl control );

    /*!
     * \brief Double precision scale dense vector by a scalar
     * \details \f$ r \leftarrow \alpha \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDscale( cldenseVector* r,
                       const clsparseScalar* alpha,
                       const cldenseVector* y,
                       const clsparseControl control );

    /*!
     * \brief Single precision scale dense vector and add dense vector
     * \details \f$ r \leftarrow \alpha \ast x + y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpy( cldenseVector* r,
                      const clsparseScalar* alpha, const cldenseVector* x,
                      const cldenseVector* y,
                      const clsparseControl control );

    /*!
     * \brief Double precision scale dense vector and add dense vector
     * \details \f$ r \leftarrow \alpha \ast x + y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value to multiply
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpy( cldenseVector* r,
                      const clsparseScalar* alpha, const cldenseVector* x,
                      const cldenseVector* y,
                      const clsparseControl control );

    /*!
     * \brief Single precision scale dense vector and add scaled dense vector
     * \details \f$ r \leftarrow \alpha \ast x + \beta \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value for x
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value for y
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSaxpby( cldenseVector* r,
                       const clsparseScalar* alpha, const cldenseVector* x,
                       const clsparseScalar* beta,
                       const cldenseVector* y,
                       const clsparseControl control );

    /*!
     * \brief Double precision scale dense vector and add scaled dense vector
     * \details \f$ r \leftarrow \alpha \ast x + \beta \ast y \f$
     * \param[out] r  Output dense vector
     * \param[in] alpha  Scalar value for x
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value for y
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDaxpby( cldenseVector* r,
                       const clsparseScalar* alpha, const cldenseVector* x,
                       const clsparseScalar* beta,
                       const cldenseVector* y,
                       const clsparseControl control );

    /*!
     * \brief Reduce integer elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseIreduce( clsparseScalar* s,
                        const cldenseVector* x,
                        const clsparseControl control );

    /*!
     * \brief Reduce single precision elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSreduce( clsparseScalar* s,
                        const cldenseVector* x,
                        const clsparseControl control );

    /*!
     * \brief Reduce double precision elements of a dense vector into a scalar value
     * \details Implicit plus operator
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDreduce( clsparseScalar* s,
                        const cldenseVector* x,
                        const clsparseControl control );

    /*!
     * \brief Calculate the single precision L1 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm1( clsparseScalar* s,
                      const cldenseVector* x,
                      const clsparseControl control );

    /*!
     * \brief Calculate the double precision L1 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm1( clsparseScalar *s,
                      const cldenseVector* x,
                      const clsparseControl control );

    /*!
     * \brief Calculate the single precision L2 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSnrm2( clsparseScalar* s,
                      const cldenseVector* x,
                      const clsparseControl control );

    /*!
     * \brief Calculate the double precision L2 norm of a dense vector
     * \param[out] s  Output scalar
     * \param[in] x  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDnrm2( clsparseScalar* s,
                      const cldenseVector* x,
                      const clsparseControl control );

    /*!
     * \brief Calculates the single precision dot-product of a dense vector
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSdot( clsparseScalar* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Calculates the double precision dot-product of a dense vector
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDdot( clsparseScalar* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

                 /* element-wise operations for dense vectors +, -, *, / */

    /*!
     * \brief Element-wise single precision addition of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSadd( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise double precision addition of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDadd( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise single precision subtraction of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSsub( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise double precision subtraction of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDsub( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise single precision multiplication of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSmul( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise double precision multiplication of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDmul( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise single precision division of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseSdiv( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );

    /*!
     * \brief Element-wise double precision division of two dense vectors
     * \param[out] r  Output dense vector
     * \param[in] x  Input dense vector
     * \param[in] y  Input dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-1
     */
    CLSPARSE_EXPORT clsparseStatus
        cldenseDdiv( cldenseVector* r,
                     const cldenseVector* x,
                     const cldenseVector* y,
                     const clsparseControl control );
    /**@}*/

    /*!
     * \defgroup BLAS-2 Sparse L2 BLAS operations
     *
     * \brief Sparse BLAS level 2 routines for sparse matrix dense vector
     * \details Level 2 BLAS operations are defined by order \f$ N^2 \f$ operations, usually
     * in the form of a matrix times a vector.
     *
     * \ingroup BLAS
     * \example sample-spmv.cpp
     */
    /**@{*/

    /*!
     * \brief Single precision CSR sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * If the CSR sparse matrix structure has rowBlocks information included,
     * then the csr-adaptive algorithm is used.  Otherwise, the csr-vector
     * algorithm is used.
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmv( const clsparseScalar* alpha,
                        const clsparseCsrMatrix* matx,
                        const cldenseVector* x,
                        const clsparseScalar* beta,
                        cldenseVector* y,
                        const clsparseControl control );

    /*!
     * \brief Double precision CSR sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * If the CSR sparse matrix structure has rowBlocks information included,
     * then the csr-adaptive algorithm is used.  Otherwise, the csr-vector
     * algorithm is used.
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrmv( const clsparseScalar* alpha,
                        const clsparseCsrMatrix* matx,
                        const cldenseVector* x,
                        const clsparseScalar* beta,
                        cldenseVector* y,
                        const clsparseControl control );


    /*!
     * \brief Single precision COO sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScoomv( const clsparseScalar* alpha,
                        const clsparseCooMatrix* matx,
                        const cldenseVector* x,
                        const clsparseScalar* beta,
                        cldenseVector* y,
                        const clsparseControl control );

    /*!
     * \brief Single precision COO sparse matrix times dense vector
     * \details \f$ y \leftarrow \alpha \ast A \ast x + \beta \ast y \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] matx  Input CSR sparse matrix
     * \param[in] x  Input dense vector
     * \param[in] beta  Scalar value to multiply against sparse vector
     * \param[out] y  Output dense vector
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup BLAS-2
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcoomv( const clsparseScalar* alpha,
                        const clsparseCooMatrix* matx,
                        const cldenseVector* x,
                        const clsparseScalar* beta,
                        cldenseVector* y,
                        const clsparseControl control );
    /**@}*/

    /*!
     * \defgroup BLAS-3 Sparse L3 BLAS operations
     *
     * \brief Sparse BLAS level 3 routines for sparse matrix dense matrix
     * \details Level 3 BLAS operations are defined by order \f$ N^3 \f$ operations,
     * usually in the form of a matrix times a matrix.
     * \ingroup BLAS
     */
    /**@{*/

    /*!
     * \brief Single precision CSR sparse matrix times dense matrix
     * \details \f$ C \leftarrow \alpha \ast A \ast B + \beta \ast C \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] sparseMatA  Input CSR sparse matrix
     * \param[in] denseMatB  Input dense matrix
     * \param[in] beta  Scalar value to multiply against dense matrix
     * \param[out] denseMatC  Output dense matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     * \note This routine is currently implemented as a batched level 2 matrix
     * times a vector.
     *
     * \ingroup BLAS-3
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsrmm( const clsparseScalar* alpha,
                        const clsparseCsrMatrix* sparseMatA,
                        const cldenseMatrix* denseMatB,
                        const clsparseScalar* beta,
                        cldenseMatrix* denseMatC,
                        const clsparseControl control );

    /*!
     * \brief Double precision CSR sparse matrix times dense matrix
     * \details \f$ C \leftarrow \alpha \ast A \ast B + \beta \ast C \f$
     * \param[in] alpha  Scalar value to multiply against sparse matrix
     * \param[in] sparseMatA  Input CSR sparse matrix
     * \param[in] denseMatB  Input dense matrix
     * \param[in] beta  Scalar value to multiply against dense matrix
     * \param[out] denseMatC  Output dense matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     * \note This routine is currently implemented as a batched level 2 matrix
     * times a vector.
     *
     * \ingroup BLAS-3
    */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsrmm( const clsparseScalar* alpha,
                        const clsparseCsrMatrix* sparseMatA,
                        const cldenseMatrix* denseMatB,
                        const clsparseScalar* beta,
                        cldenseMatrix* denseMatC,
                        const clsparseControl control );
    /**@}*/

    /*!
     * \defgroup CONVERT Matrix conversion routines
     *
     * \brief Sparse matrix routines to convert from one format into another
     * \note Input sparse matrices have to be sorted by row first and then column.
     * The sparse conversion routines below require this property, and the clsparse
     * matrix file reading routines clsparse?C??MatrixfromFile guarantee that property
     */
    /**@{*/

    /*!
     * \brief Convert a single precision CSR encoded sparse matrix into a COO encoded sparse matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] coo  Output COO encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsr2coo( const clsparseCsrMatrix* csr,
                          clsparseCooMatrix* coo,
                          const clsparseControl control );

    /*!
     * \brief Convert a double precision CSR encoded sparse matrix into a COO encoded sparse matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] coo  Output COO encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsr2coo( const clsparseCsrMatrix* csr,
                          clsparseCooMatrix* coo,
                          const clsparseControl control );

    /*!
     * \brief Convert a single precision COO encoded sparse matrix into a CSR encoded sparse matrix
     * \param[in] coo  Input COO encoded sparse matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScoo2csr( const clsparseCooMatrix* coo,
                          clsparseCsrMatrix* csr,
                          const clsparseControl control );

    /*!
     * \brief Convert a double precision COO encoded sparse matrix into a CSR encoded sparse matrix
     * \param[in] coo  Input COO encoded sparse matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcoo2csr( const clsparseCooMatrix* coo,
                          clsparseCsrMatrix* csr,
                          const clsparseControl control );

    /*!
     * \brief Convert a single precision CSR encoded sparse matrix into a dense matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] A  Output dense matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseScsr2dense( const clsparseCsrMatrix* csr,
                            cldenseMatrix* A,
                            const clsparseControl control );

    /*!
     * \brief Convert a double precision CSR encoded sparse matrix into a dense matrix
     * \param[in] csr  Input CSR encoded sparse matrix
     * \param[out] A  Output dense matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDcsr2dense( const clsparseCsrMatrix* csr,
                            cldenseMatrix* A,
                            clsparseControl control );

    /*!
     * \brief Convert a single precision dense matrix into a CSR encoded sparse matrix
     * \param[in] A  Input dense matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseSdense2csr( const cldenseMatrix* A,
                            clsparseCsrMatrix* csr,
                            const clsparseControl control );

    /*!
     * \brief Convert a double precision dense matrix into a CSR encoded sparse matrix
     * \param[in] A  Input dense matrix
     * \param[out] csr  Output CSR encoded sparse matrix
     * \param[in] control A valid clsparseControl created with clsparseCreateControl
     *
     * \ingroup CONVERT
     */
    CLSPARSE_EXPORT clsparseStatus
        clsparseDdense2csr( const cldenseMatrix* A, clsparseCsrMatrix* csr,
                            const clsparseControl control );
    /**@}*/

  /*!
   * \brief Single Precision CSR Sparse Matrix times Sparse Matrix
   * \details \f$ C \leftarrow A \ast B \f$
   * \warning The column index of each row of CSR matrixes must be ordered (sorted)
   * \param[in] sparseMatA Input CSR sparse matrix
   * \param[in] sparseMatB Input CSR sparse matrix
   * \param[out] sparseMatC Output CSR sparse matrix
   * \param[in] control A valid clsparseControl created with clsparseCreateControl
   *
   * \ingroup BLAS-3
   */
 CLSPARSE_EXPORT clsparseStatus
        clsparseScsrSpGemm(
        const clsparseCsrMatrix* sparseMatA,
        const clsparseCsrMatrix* sparseMatB,
              clsparseCsrMatrix* sparseMatC,
        const clsparseControl control );


#ifdef __cplusplus
}      // extern C
#endif

#endif // _CL_SPARSE_H_
