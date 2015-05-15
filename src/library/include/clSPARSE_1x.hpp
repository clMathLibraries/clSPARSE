#pragma once
#ifndef _CL_SPARSE_1x_HPP_
#define _CL_SPARSE_1x_HPP_

#include <iostream>
#include <type_traits>
#include "clSPARSE_1x.h"

// C++ wrapper classes that inherit from the extenrally visible C classes, 
// for the purpose of providing convenience methods to abstract away the 
// differences between cl1.2 and cl2.0
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and 
// destroying these objects

//inline cl_mem clAllocateMem( cl_context cl_ctx, size_t size, cl_mem_flags flags, void* hostBuffer )
//{
//    cl_mem buf;
//    cl_int status;
//
//    buf = clCreateBuffer( cl_ctx, flags, size, hostBuffer, &status );
//
//    return buf;
//}

template< typename pType >
class clMemRAII
{
    cl_command_queue clQueue;
    cl_mem clBuff;
    pType* clMem;

public:

    clMemRAII() : clQueue(nullptr), clBuff(nullptr), clMem(nullptr)
    {

    }

    clMemRAII( const cl_command_queue cl_queue, const cl_mem cl_buff,
               const size_t cl_size = 0, const cl_mem_flags cl_flags = CL_MEM_READ_WRITE) :
        clMem( nullptr )
    {
        clQueue = cl_queue;
        clBuff = cl_buff;

        ::clRetainCommandQueue( clQueue );
         if(cl_size > 0)
         {
             cl_context ctx = NULL;

             ::clGetCommandQueueInfo(clQueue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL);
             cl_int status = 0;
             clBuff = ::clCreateBuffer(ctx, cl_flags, cl_size * sizeof(pType), NULL, &status);
         }
         else
         {
            ::clRetainMemObject( clBuff );
         }
    }

    pType* clMapMem( cl_bool clBlocking, const cl_map_flags clFlags, const size_t clOff, const size_t clSize )
    {
        // Right now, we don't support returning an event to wait on
        clBlocking = CL_TRUE;
        cl_int clStatus = 0;

        clMem = static_cast< pType* >( ::clEnqueueMapBuffer( clQueue, clBuff, clBlocking, clFlags, clOff, 
            clSize * sizeof( pType ), 0, NULL, NULL, &clStatus ) );

        return clMem;
    }

    void clWriteMem( cl_bool clBlocking, const size_t clOff, const size_t clSize, const void* hostMem )
    {
        // Right now, we don't support returning an event to wait on
        clBlocking = CL_TRUE;

        cl_int clStatus = ::clEnqueueWriteBuffer( clQueue, clBuff, clBlocking, clOff,
                                                  clSize * sizeof( pType ), hostMem, 0, NULL, NULL );
    }

    ~clMemRAII( )
    {
        if( clMem )
            ::clEnqueueUnmapMemObject( clQueue, clBuff, clMem, 0, NULL, NULL );

        ::clReleaseCommandQueue( clQueue );

        ::clReleaseMemObject( clBuff );

    }
};

class clsparseScalarPrivate: public clsparseScalar
{
public:

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
        offValues = offColInd = offRowOff = offRowBlocks = rowBlockSize = 0;
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

    cl_ulong rowBlocksOffset( ) const
    {
        return offRowBlocks;
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

    cl_uint nnz_per_row( ) const
    {
        return nnz / m;
    }

    cl_ulong valOffset( ) const
    {
        return offValues;
    }

    cl_ulong colIndOffset( ) const
    {
        return offColInd;
    }

    cl_ulong rowOffOffset( ) const
    {
        return offRowInd;
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
