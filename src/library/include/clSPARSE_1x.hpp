#pragma once
#ifndef _CL_SPARSE_1x_HPP_
#define _CL_SPARSE_1x_HPP_

#include <type_traits>
#include "clSPARSE_1x.h"

// C++ wrapper classes that inherit from the extenrally visible C classes, 
// for the purpose of providing convenience methods to abstract away the 
// differences between cl1.2 and cl2.0
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and 
// destroying these objects
class clsparseScalarPrivate: public clsparseScalar
{
public:
    typedef cl_mem value_type;

    void clear( )
    {
        value = nullptr;
        offValue = 0;
    }

    cl_ulong offset () const
    {
        return offValue;
    }
};

class clsparseVectorPrivate: public clsparseVector
{
public:
    void clear( )
    {
        n = 0;
        values = nullptr;
        offValues = 0;
    }

    cl_ulong offset () const
    {
        return offValues;
    }
};

class clsparseCsrMatrixPrivate: public clsparseCsrMatrix
{
public:
    void clear( )
    {
        m = n = nnz = 0;
        values = colIndices = rowOffsets = rowBlocks = nullptr;
        offValues = offColInd = offRowOff = offRowBlocks = 0;
    }

    cl_uint nnz_per_row() const
    {
        return nnz/m;
    }

    cl_ulong valOffset () const
    {
        return offValues;
    }

    cl_ulong colIndOffset () const
    {
        return offColInd;
    }

    cl_ulong rowOffOffset () const
    {
        return offRowOff;
    }


};

class clsparseCooMatrixPrivate: public clsparseCooMatrix
{
public:
    void clear( )
    {
        m = n = nnz = 0;
        values = colIndices = rowIndices = nullptr;
        offValues = offColInd = offRowInd = 0;
    }
};

class clsparseDenseMatrixPrivate: public clsparseDenseMatrix
{
public:
    void clear( )
    {
        m = n = 0;
        values = nullptr;
    }
};

// Check that it is OK to static_cast a C struct pointer to a C++ class pointer
static_assert( std::is_standard_layout< clsparseScalarPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseVectorPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCsrMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCooMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseDenseMatrixPrivate>::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );

#endif
