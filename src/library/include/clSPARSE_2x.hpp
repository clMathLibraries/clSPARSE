#pragma once
#ifndef _CL_SPARSE_2x_HPP_
#define _CL_SPARSE_2x_HPP_

#include "clSPARSE_2x.h"

// C++ wrapper classes that inherit from the extenrally visible C classes, 
// for the purpose of providing convenience methods to abstract away the 
// differences between cl1.2 and cl2.0
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and 
// destroying these objects
class clsparseScalarPrivate : public clsparseScalar
{
public:
    void clear( )
    {
        value = nullptr;
    }
};

class clsparseVectorPrivate: public clsparseVector
{
public:
    void clear( )
    {
        n = 0;
        values = nullptr;
    }
};

class clsparseCsrMatrixPrivate: public clsparseCsrMatrix
{
public:
    void clear( )
    {
        m = n = nnz = 0;
        values = colIndices = rowOffsets = rowBlocks = nullptr;
    }
};

class clsparseCooMatrixPrivate: public clsparseCooMatrix
{
public:
    void clear( )
    {
        m = n = nnz = 0;
        values = colIndices = rowIndices = nullptr;
    }
};

#endif