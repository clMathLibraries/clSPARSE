/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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
 * \brief clSPARSE-1x.h defines public types specific to OpenCL 1.x API's.
 * This file is kept as a strictly 'C' compatible interface.
 */

#pragma once
#ifndef _CL_SPARSE_1x_H_
#define _CL_SPARSE_1x_H_

#include "clSPARSE-xx.h"


/*! \brief Structure to encapsulate scalar data to clSPARSE API
 */
typedef struct clsparseScalar_
{
    cl_mem value;  /**< OpenCL 1.x memory handle */

    /*! Given that cl_mem objects are opaque without pointer arithmetic, this offset is added to
     * the cl_mem locations on device to define beginning of the data in the cl_mem buffers
     */
    cl_ulong offValue;
} clsparseScalar;

/*! \brief Structure to encapsulate dense vector data to clSPARSE API
 */
typedef struct cldenseVector_
{
    clsparseIdx_t num_values;  /*!< Length of dense vector */

    cl_mem values;  /*!< OpenCL 1.x memory handle */

    /*! Given that cl_mem objects are opaque without pointer arithmetic, this offset is added to
     * the cl_mem locations on device to define beginning of the data in the cl_mem buffers
     */
    cl_ulong offValues;
} cldenseVector;

/*! \brief Structure to encapsulate sparse matrix data encoded in CSR
 * form to clSPARSE API
 * \note The indices stored are 0-based
 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 */
typedef struct clsparseCsrMatrix_
{
    /** @name CSR matrix data */
    /**@{*/
    clsparseIdx_t num_rows;  /*!< Number of rows this matrix has if viewed as dense */
    clsparseIdx_t num_cols;  /*!< Number of columns this matrix has if viewed as dense */
    clsparseIdx_t num_nonzeros;  /*!< Number of values in matrix that are non-zero */
    /**@}*/

    /** @name OpenCL state */
    /**@{*/
    cl_mem values;  /*!< non-zero values in sparse matrix of size num_nonzeros */
    cl_mem colIndices;  /*!< column index for corresponding value of size num_nonzeros */
    cl_mem rowOffsets;  /*!< Invariant: rowOffsets[i+1]-rowOffsets[i] = number of values in row i */
    cl_mem rowBlocks;  /*!< Meta-data used for csr-adaptive algorithm; can be NULL */
    /**@}*/

    /** @name Buffer offsets */
    /**@{*/
    /*! Given that cl_mem objects are opaque without pointer arithmetic, these offsets are added to
     * the cl_mem locations on device to define beginning of the data in the cl_mem buffers
     */
    cl_ulong offValues;
    cl_ulong offColInd;
    cl_ulong offRowOff;
    cl_ulong offRowBlocks;
    /**@}*/

    size_t rowBlockSize;  /*!< Size of array used by the rowBlocks handle */
} clsparseCsrMatrix;

/*! \brief Structure to encapsulate sparse matrix data encoded in COO
 * form to clSPARSE API
 * \note The indices stored are 0-based
 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 */
typedef struct clsparseCooMatrix_
{
    /** @name COO matrix data */
    /**@{*/
    clsparseIdx_t num_rows;  /*!< Number of rows this matrix has if viewed as dense */
    clsparseIdx_t num_cols;  /*!< Number of columns this matrix has if viewed as dense */
    clsparseIdx_t num_nonzeros;  /*!< Number of values in matrix that are non-zero */
    /**@}*/

    /** @name OpenCL state */
    /**@{*/
    cl_mem values;  /*!< CSR non-zero values of size num_nonzeros */
    cl_mem colIndices;  /*!< column index for corresponding element; array size num_nonzeros */
    cl_mem rowIndices;  /*!< row index for corresponding element; array size num_nonzeros */
    /**@}*/

    /** @name Buffer offsets */
    /**@{*/
    /*! Given that cl_mem objects are opaque without pointer arithmetic, these offsets are added to
    * the cl_mem locations on device to define beginning of the data in the cl_mem buffers
    */
    cl_ulong offValues;
    cl_ulong offColInd;
    cl_ulong offRowInd;
    /**@}*/
} clsparseCooMatrix;

/*! \brief Structure to encapsulate dense matrix data to clSPARSE API
 * \note It is the users responsibility to allocate/deallocate OpenCL buffers
 */
typedef struct cldenseMatrix_
{
    /** @name Dense matrix data */
    /**@{*/
    clsparseIdx_t num_rows;  /*!< Number of rows */
    clsparseIdx_t num_cols;  /*!< Number of columns */
    clsparseIdx_t lead_dim;  /*! Stride to the next row or column, in units of elements */
    cldenseMajor major;  /*! Memory layout for dense matrix */
    /**@}*/

    cl_mem values;  /*!< Array of matrix values */

    /*! Given that cl_mem objects are opaque without pointer arithmetic, these offsets are added to
    * the cl_mem locations on device to define beginning of the data in the cl_mem buffers
    */
    cl_ulong offValues;
} cldenseMatrix;

#endif
