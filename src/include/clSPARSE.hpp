#pragma once
#ifndef _CL_SPARSE_HPP_
#define _CL_SPARSE_HPP_

//Include all of our clSPARSE API
#include "clSPARSE.h"

class clAllocator
{
public:
    cl_command_queue queue;

    virtual void* operator( )( size_t buffSize ) const = 0;
};

// clAllocator is a functor or lambda expression that this function uses to allocate memory for the 
// buffers contained in clsparseCooMatrix
CLSPARSE_EXPORT clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, const clAllocator& clAlloc );

#endif // _CL_SPARSE_HPP_
