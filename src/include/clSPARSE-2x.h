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

#pragma once
#ifndef _CL_SPARSE_2x_H_
#define _CL_SPARSE_2x_H_

#include "clSPARSE-xx.h"

// Data types used to pass OpenCL objects into the clSPARSE library
// These are plain PoD containers; no methods defined
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and 
// destroying these objects
typedef struct clsparseScalar_
{
    // OpenCL state
    void* value;
} clsparseScalar;

typedef struct cldenseVector_
{
    // Matrix meta
    cl_int num_values;

    // OpenCL state
    void* values;
} cldenseVector;

typedef struct clsparseCsrMatrix_
{
    // Matrix meta
    cl_int num_rows;
    cl_int num_cols;
    cl_int num_nonzeros;

    // OpenCL state
    void* values;
    void* colIndices;
    void* rowOffsets;
    void* rowBlocks;      // It is possible that this pointer may be NULL

    size_t rowBlockSize;
} clsparseCsrMatrix;

typedef struct clsparseCooMatrix_
{
    // Matrix meta
    cl_int num_rows;
    cl_int num_cols;
    cl_int num_nonzeros;

    // OpenCL state
    void* values;
    void* colIndices;
    void* rowIndices;
} clsparseCooMatrix;

typedef struct cldenseMatrix_
{
    size_t num_rows;
    size_t num_cols;
    size_t lead_dim;
    cldenseMajor major;

    void* values;

} cldenseMatrix;

#endif
