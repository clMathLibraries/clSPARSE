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
#ifndef _CLSPARSE_CSRMV_ADAPTIVE_HPP_
#define _CLSPARSE_CSRMV_ADAPTIVE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-validate.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "internal/data-types/clvector.hpp"

template <typename T>
clsparseStatus
csrmv_adaptive( const clsparseScalarPrivate* pAlpha,
                const clsparseCsrMatrixPrivate* pCsrMatx,
                const cldenseVectorPrivate* pX,
                const clsparseScalarPrivate* pBeta,
                cldenseVectorPrivate* pY,
                clsparseControl control )
{


    const cl_uint group_size = 256;

    std::string params = std::string( )
    + " -DINDEX_TYPE=uint"
    + " -DROWBITS=" + std::to_string( ROW_BITS )
    + " -DWGBITS=" + std::to_string( WG_BITS )
    + " -DWG_SIZE=" + std::to_string( group_size )
    + " -DBLOCKSIZE=" + std::to_string( BLKSIZE )
    + " -DBLOCK_MULTIPLIER=" + std::to_string( BLOCK_MULTIPLIER )
    + " -DROWS_FOR_VECTOR=" + std::to_string( ROWS_FOR_VECTOR )
    + " -DEXTENDED_PRECISION";

    std::string options;
    if(typeid(T) == typeid(cl_double))
        options = std::string() + " -DVALUE_TYPE=double -DDOUBLE";
    else if(typeid(T) == typeid(cl_float))
        options = std::string() + " -DVALUE_TYPE=float";
    else if(typeid(T) == typeid(cl_uint))
        options = std::string() + " -DVALUE_TYPE=uint";
    else if(typeid(T) == typeid(cl_int))
        options = std::string() + " -DVALUE_TYPE=int";
    else if(typeid(T) == typeid(cl_ulong))
        options = std::string() + " -DVALUE_TYPE=ulong -DLONG";
    else if(typeid(T) == typeid(cl_long))
        options = std::string() + " -DVALUE_TYPE=long -DLONG";
    else
        return clsparseInvalidKernelArgs;
    params.append(options);

    cl::Kernel kernel = KernelCache::get( control->queue,
                                          "csrmv_adaptive",
                                          "csrmv_adaptive",
                                          params );

    KernelWrap kWrapper( kernel );

    kWrapper << pCsrMatx->values
        << pCsrMatx->colIndices << pCsrMatx->rowOffsets
        << pX->values << pY->values
        << pCsrMatx->rowBlocks
        << pAlpha->value << pBeta->value;
        //<< h_alpha << h_beta;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    // Setting global work size to half the row block size because we are only
    // using half the row blocks buffer for actual work.
    // The other half is used for the extended precision reduction.
    cl_uint global_work_size = ( (pCsrMatx->rowBlockSize/2) - 1 ) * group_size;
    cl::NDRange local( group_size );
    cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

    cl_int status = kWrapper.run( control, global, local );

    if( status != CL_SUCCESS )
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}


/*
 * clsparse::array
 */

template <typename T>
clsparseStatus
csrmv_adaptive( const clsparse::array_base<T>& pAlpha,
                const clsparseCsrMatrixPrivate* pCsrMatx,
            const clsparse::array_base<T>& pX,
            const clsparse::array_base<T>& pBeta,
            clsparse::array_base<T>& pY,
            clsparseControl control )
{

    const cl_uint group_size = 256;

    std::string params = std::string( )
    + " -DINDEX_TYPE=uint"
    + " -DROWBITS=" + std::to_string( ROW_BITS )
    + " -DWGBITS=" + std::to_string( WG_BITS )
    + " -DWG_SIZE=" + std::to_string( group_size )
    + " -DBLOCKSIZE=" + std::to_string( BLKSIZE )
    + " -DBLOCK_MULTIPLIER=" + std::to_string( BLOCK_MULTIPLIER )
    + " -DROWS_FOR_VECTOR=" + std::to_string( ROWS_FOR_VECTOR )
    + " -DEXTENDED_PRECISION";

    std::string options;
    if(typeid(T) == typeid(cl_double))
        options = std::string() + " -DVALUE_TYPE=double -DDOUBLE";
    else if(typeid(T) == typeid(cl_float))
        options = std::string() + " -DVALUE_TYPE=float";
    else if(typeid(T) == typeid(cl_uint))
        options = std::string() + " -DVALUE_TYPE=uint";
    else if(typeid(T) == typeid(cl_int))
        options = std::string() + " -DVALUE_TYPE=int";
    else if(typeid(T) == typeid(cl_ulong))
        options = std::string() + " -DVALUE_TYPE=ulong -DLONG";
    else if(typeid(T) == typeid(cl_long))
        options = std::string() + " -DVALUE_TYPE=long -DLONG";
    else
        return clsparseInvalidKernelArgs;
    params.append(options);

    cl::Kernel kernel = KernelCache::get( control->queue,
                                          "csrmv_adaptive",
                                          "csrmv_adaptive",
                                          params );

    KernelWrap kWrapper( kernel );

    kWrapper << pCsrMatx->values
        << pCsrMatx->colIndices << pCsrMatx->rowOffsets
        << pX.data() << pY.data()
        << pCsrMatx->rowBlocks
        << pAlpha.data() << pBeta.data();
        //<< h_alpha << h_beta;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    // Setting global work size to half the row block size because we are only
    // using half the row blocks buffer for actual work.
    // The other half is used for the extended precision reduction.
    cl_uint global_work_size = ( (pCsrMatx->rowBlockSize/2) - 1 ) * group_size;
    cl::NDRange local( group_size );
    cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

    cl_int status = kWrapper.run( control, global, local );

    if( status != CL_SUCCESS )
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_CSRMV_ADAPTIVE_HPP_
