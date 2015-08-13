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

#pragma once
#ifndef _CL_SPARSE_1x_HPP_
#define _CL_SPARSE_1x_HPP_

#include <iostream>
#include <type_traits>
#include "clSPARSE-1x.h"
#include "clSPARSE-error.h"

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

//    clMemRAII() : clQueue(nullptr), clBuff(nullptr), clMem(nullptr)
//    {

//    }

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

    clMemRAII( const cl_command_queue cl_queue, cl_mem* cl_buff,
               const size_t cl_size = 0, const cl_mem_flags cl_flags = CL_MEM_READ_WRITE) :
        clMem( nullptr )
    {
        clQueue = cl_queue;
        clBuff = *cl_buff;

        ::clRetainCommandQueue( clQueue );
         if(cl_size > 0)
         {
             cl_context ctx = NULL;

             ::clGetCommandQueueInfo(clQueue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL);
             cl_int status = 0;
             clBuff = ::clCreateBuffer(ctx, cl_flags, cl_size * sizeof(pType), NULL, &status);
             *cl_buff = clBuff;
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

    //simple fill mem wrapper, pattern is a single value for now
    void clFillMem (const pType pattern, const size_t clOff, const size_t clSize)
    {
        cl_int clStatus = clEnqueueFillBuffer(clQueue, clBuff,
                                              &pattern, sizeof(pType), clOff,
                                              clSize * sizeof(pType),
                                              0, NULL, NULL);
    }

    ~clMemRAII( )
    {
	cl_int clStatus = 0;

        if( clMem )
	{
	    cl_event unmapEvent = nullptr;
            clStatus = ::clEnqueueUnmapMemObject( clQueue, clBuff, clMem, 0, NULL, &unmapEvent );
	    CLSPARSE_V( clStatus, "::clEnqueueUnmapMemObject" );
            clStatus = ::clWaitForEvents( 1, &unmapEvent );
	    CLSPARSE_V( clStatus, "::clWaitForEvents" );
            clStatus = ::clReleaseEvent( unmapEvent );
	}

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

class cldenseVectorPrivate: public cldenseVector
{
public:
    void clear( )
    {
        num_values = 0;
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
        num_rows = num_cols = num_nonzeros = 0;
        values = colIndices = rowOffsets = rowBlocks = nullptr;
        offValues = offColInd = offRowOff = offRowBlocks = rowBlockSize = 0;
    }

    cl_uint nnz_per_row() const
    {
        return num_nonzeros / num_rows;
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
        num_rows = num_cols = num_nonzeros = 0;
        values = colIndices = rowIndices = nullptr;
        offValues = offColInd = offRowInd = 0;
    }

    cl_uint nnz_per_row( ) const
    {
        return num_nonzeros / num_rows;
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

class cldenseMatrixPrivate: public cldenseMatrix
{
public:
    void clear( )
    {
        num_rows = num_cols = lead_dim = offValues = 0;
        major = rowMajor;
        values = nullptr;
    }

    cl_ulong offset( ) const
    {
        return offValues;
    }

};

// Check that it is OK to static_cast a C struct pointer to a C++ class pointer
static_assert( std::is_standard_layout< clsparseScalarPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< cldenseVectorPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCsrMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCooMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< cldenseMatrixPrivate>::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );

#endif
