#pragma once
#ifndef _CL_SPARSE_1x_H_
#define _CL_SPARSE_1x_H_

#include "clSPARSE-xx.h"

// Data types used to pass OpenCL objects into the clSPARSE library
// These are plain PoD containers; no methods defined
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and 
// destroying these objects
typedef struct clsparseScalar_
{
    // OpenCL state
    cl_mem value;

    //OpenCL meta
    cl_ulong offValue;
} clsparseScalar;

typedef struct cldenseVector_
{
    // Matrix meta
    cl_int n;

    // OpenCL state
    cl_mem values;

    //OpenCL meta
    cl_ulong offValues;
} cldenseVector;

typedef struct clsparseCsrMatrix_
{
    // Matrix meta
    cl_int num_rows;
    cl_int num_cols;
    cl_int num_nonzeros;

    // OpenCL state
    cl_mem values;
    cl_mem colIndices;
    cl_mem rowOffsets;
    cl_mem rowBlocks;      // It is possible that this pointer may be NULL

    //OpenCL meta
    cl_ulong offValues;
    cl_ulong offColInd;
    cl_ulong offRowOff;
    cl_ulong offRowBlocks;
    size_t rowBlockSize;
} clsparseCsrMatrix;

typedef struct clsparseCooMatrix_
{
    // Matrix meta
    cl_int num_rows;
    cl_int num_cols;
    cl_int num_nonzeros;

    // OpenCL state
    cl_mem values;
    cl_mem colIndices;
    cl_mem rowIndices;

    //OpenCL meta
    cl_ulong offValues;
    cl_ulong offColInd;
    cl_ulong offRowInd;
} clsparseCooMatrix;

//for sake of clarity in the interface
typedef struct cldenseMatrix_
{
    size_t num_rows;
    size_t num_cols;
    size_t lead_dim;
    cldenseMajor major;

    cl_mem values;

    //OpenCL meta
    cl_ulong offValues;

} cldenseMatrix;

#endif
