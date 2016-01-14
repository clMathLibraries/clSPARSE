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
#ifndef _CL_SPARSE_2x_HPP_
#define _CL_SPARSE_2x_HPP_

#include <type_traits>
#include "clSPARSE-2x.h"

// C++ wrapper classes that inherit from the externally visible C classes,
// for the purpose of providing convenience methods to abstract away the
// differences between cl1.2 and cl2.0
// Users are responsible for creating and destroying the OpenCL objects
// Helper functions may be provided to assist users in creating and
// destroying these objects

//inline void* clAllocateMem( cl_context cl_ctx, size_t size, cl_svm_mem_flags flags, void* hostBuffer )
//{
//    void* buf;
//    cl_int status;
//
//    buf = clSVMAlloc( cl_ctx, flags, size, 0 );
//
//    return buf;
//}

template< typename pType >
class clMemRAII
{
    cl_command_queue clQueue;
    pType* clMem;
    cl_bool clOwner;

public:

    //temporary solution for situations when clMemRaII is allocating buffer,
    // bug should not release it when calling the destructor.
    // ProperFIX: write operator=(const clMemRAII&)
    clMemRAII( const cl_command_queue cl_queue, void** cl_malloc,
               const size_t cl_size = 0, const cl_svm_mem_flags cl_flags = CL_MEM_READ_WRITE):
        clMem( nullptr ), clOwner(false)
    {
        clQueue = cl_queue;
        clMem = static_cast< pType* >( *cl_malloc );

        if(cl_size > 0)
        {
            cl_context ctx = NULL;

            ::clGetCommandQueueInfo(clQueue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL);
            cl_int status = 0;

            clMem = static_cast< pType* > (clSVMAlloc(ctx, cl_flags, cl_size * sizeof(pType), 0));
            *cl_malloc = clMem;
        }

        ::clRetainCommandQueue( clQueue );
    }

    clMemRAII( const cl_command_queue cl_queue, void* cl_malloc,
               const size_t cl_size = 0, const cl_svm_mem_flags cl_flags = CL_MEM_READ_WRITE):
        clMem( nullptr ), clOwner(false)
    {
        clQueue = cl_queue;
        clMem = static_cast< pType* >( cl_malloc );

        if(cl_size > 0)
        {
            cl_context ctx = NULL;

            ::clGetCommandQueueInfo(clQueue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL);
            cl_int status = 0;

            clMem = static_cast< pType* > (clSVMAlloc(ctx, cl_flags, cl_size * sizeof(pType), 0));
            clOwner = true;
        }

        ::clRetainCommandQueue( clQueue );
    }

    pType* clMapMem( cl_bool clBlocking, const cl_map_flags clFlags, const size_t clOff, const size_t clSize, cl_int *clStatus = nullptr)
    {
        // Right now, we don't support returning an event to wait on
        clBlocking = CL_TRUE;

        cl_int _clStatus = ::clEnqueueSVMMap( clQueue, clBlocking, clFlags,
                                              clMem, clSize * sizeof( pType ), 0, NULL, NULL );
        if (clStatus != nullptr)
        {
            *clStatus = _clStatus;
        }

        return clMem;
    }

    void clWriteMem( cl_bool clBlocking, const size_t clOff, const size_t clSize, const void* srcPtr )
    {
        // Right now, we don't support returning an event to wait on
        clBlocking = CL_TRUE;

        cl_int clStatus = ::clEnqueueSVMMemcpy( clQueue, clBlocking, clMem, srcPtr,
                                                  clSize * sizeof( pType ), 0, NULL, NULL );
    }

    void clFillMem (const pType pattern, const size_t clOff, const size_t clSize)
    {
        cl_int clStatus = ::clEnqueueSVMMemFill(clQueue, clMem,
                                                &pattern, sizeof(pType),
                                                clSize * sizeof(pType),
                                                0, NULL, NULL);
    }

    ~clMemRAII( )
    {
        if( clMem )
            ::clEnqueueSVMUnmap( clQueue, clMem, 0, NULL, NULL );

        if(clOwner)
        {
            cl_context ctx = nullptr;
            ::clGetCommandQueueInfo( clQueue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL);
            ::clSVMFree(ctx, clMem);
        }

        ::clReleaseCommandQueue( clQueue );
    }
};

class clsparseScalarPrivate : public clsparseScalar
{
public:
    void clear( )
    {
        value = nullptr;
    }

    cl_ulong offset () const
    {
        return 0;
    }

};

class cldenseVectorPrivate: public cldenseVector
{
public:
    void clear( )
    {
        num_values = 0;
        values = nullptr;
    }

    cl_ulong offset () const
    {
        return 0;
    }

};

class clsparseCsrMatrixPrivate: public clsparseCsrMatrix
{
public:
    void clear( )
    {
        num_rows = num_cols = num_nonzeros = 0;
        values = col_indices = row_pointer = rowBlocks = nullptr;
        rowBlockSize = 0;
    }

    clsparseIdx_t nnz_per_row() const
    {
        return num_nonzeros / num_rows;
    }

    cl_ulong valOffset () const
    {
        return 0;
    }

    cl_ulong colIndOffset () const
    {
        return 0;
    }

    cl_ulong rowOffOffset () const
    {
        return 0;
    }

    cl_ulong rowBlocksOffset( ) const
    {
        return 0;
    }
};

class clsparseCooMatrixPrivate: public clsparseCooMatrix
{
public:
    void clear( )
    {
        num_rows = num_cols = num_nonzeros = 0;
        values = col_indices = row_indices = nullptr;
    }

    clsparseIdx_t nnz_per_row( ) const
    {
        return num_nonzeros / num_rows;
    }

    cl_ulong valOffset( ) const
    {
        return 0;
    }

    cl_ulong colIndOffset( ) const
    {
        return 0;
    }

    cl_ulong rowOffOffset( ) const
    {
        return 0;
    }
};

class cldenseMatrixPrivate: public cldenseMatrix
{
public:
    void clear( )
    {
        num_rows = num_cols = lead_dim = 0;
        major = rowMajor;
        values = nullptr;
    }
};

// Check that it is OK to static_cast a C struct pointer to a C++ class pointer
static_assert( std::is_standard_layout< clsparseScalarPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< cldenseVectorPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCsrMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< clsparseCooMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );
static_assert( std::is_standard_layout< cldenseMatrixPrivate >::value, "The C++ wrapper classes have to have same memory layout as the C class they inherit from" );

#endif
