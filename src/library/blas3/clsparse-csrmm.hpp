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
#ifndef _CLSPARSE_CSRMM_HPP_
#define _CLSPARSE_CSRMM_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"

//template <typename T>
//clsparseStatus
//csrmm_adaptive( const clsparseScalarPrivate& pAlpha,
//const clsparseCsrMatrixPrivate& pSparseCsrA,
//const cldenseMatrixPrivate& pDenseB,
//const clsparseScalarPrivate& pBeta,
//cldenseMatrixPrivate& pDenseC,
//const clsparseControl control )
//{
//  if( typeid( T ) == typeid( cl_double ) )
//  {
//      return clsparseNotImplemented;
//  }
//
//  if( ( pDenseB.major != rowMajor ) && ( pDenseC.major != rowMajor ) )
//  {
//      return clsparseNotImplemented;
//  }
//
//  const cl_uint group_size = 256;
//
//  const std::string params = std::string( )
//  + " -DROWBITS=" + std::to_string( ROW_BITS )
//  + " -DWGBITS=" + std::to_string( WG_BITS )
//  + " -DBLOCKSIZE=" + std::to_string( BLKSIZE );
//#ifdef DOUBLE
//  buildFlags += " -DDOUBLE";
//#endif
//
//  cl::Kernel kernel = KernelCache::get( control->queue,
//                                        "csrmm",
//                                        "csrmm_ulong",
//                                        params );
//
//  KernelWrap kWrapper( kernel );
//
//  kWrapper << pSparseCsrA.values << pSparseCsrA.col_indices << pSparseCsrA.row_pointer << pSparseCsrA.rowBlocks
//      << pDenseB.values << pDenseB.lead_dim
//      << pDenseC.values << pDenseC.num_rows << pDenseC.num_cols << pDenseC.lead_dim
//      << pAlpha.value << pBeta.value;
//
//  // if NVIDIA is used it does not allow to run the group size
//  // which is not a multiplication of group_size. Don't know if that
//  // have an impact on performance
//  cl_uint global_work_size = ( pSparseCsrA.rowBlockSize - 1 ) * group_size;
//  cl::NDRange local( group_size );
//  cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );
//
//  cl_int status = kWrapper.run( control, global, local );
//
//  if( status != CL_SUCCESS )
//  {
//      return clsparseInvalidKernelExecution;
//  }
//
//  return clsparseSuccess;
//}

template<typename T>
clsparseStatus
csrmm( const clsparseScalarPrivate& pAlpha,
const clsparseCsrMatrixPrivate& pSparseCsrA,
const cldenseMatrixPrivate& pDenseB,
const clsparseScalarPrivate& pBeta,
cldenseMatrixPrivate& pDenseC,
const clsparseControl control )
{
    clsparseIdx_t nnz_per_row = pSparseCsrA.nnz_per_row(); //average nnz per row
    clsparseIdx_t wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    clsparseIdx_t subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if( wave_size > 32 )
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if( nnz_per_row < 64 ) { subwave_size = 32; }
    }
    if( nnz_per_row < 32 ) { subwave_size = 16; }
    if( nnz_per_row < 16 ) { subwave_size = 8; }
    if( nnz_per_row < 8 )  { subwave_size = 4; }
    if( nnz_per_row < 4 )  { subwave_size = 2; }

    std::string params = std::string( ) +
        + " -DVALUE_TYPE=" + OclTypeTraits<T>::type        
        + " -DWG_SIZE=" + std::to_string( group_size )
        + " -DWAVE_SIZE=" + std::to_string( wave_size )
        + " -DSUBWAVE_SIZE=" + std::to_string( subwave_size );

    if (sizeof(clsparseIdx_t) == 8)
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type;
        params.append(options);
    }
    else
    {
        std::string options = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_uint>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_uint>::type;
        params.append(options);
    }

    if( typeid( T ) == typeid( cl_double ) )
    {
        params += " -DDOUBLE";
        if (!control->dpfp_support)
        {
#ifndef NDEBUG
            std::cerr << "Failure attempting to run double precision kernel on device without DPFP support." << std::endl;
#endif
            return clsparseInvalidDevice;
        }
    }

    cl::Kernel kernel = KernelCache::get( control->queue,
                                          "csrmm_general",
                                          "csrmv_batched",
                                          params );
    KernelWrap kWrapper( kernel );

    kWrapper << pSparseCsrA.num_rows
        << pAlpha.value << pAlpha.offset( )
        << pSparseCsrA.row_pointer << pSparseCsrA.col_indices << pSparseCsrA.values
        << pDenseB.values << pDenseB.lead_dim << pDenseB.offset( )
        << pBeta.value << pBeta.offset( )
        << pDenseC.values << pDenseC.num_rows << pDenseC.num_cols << pDenseC.lead_dim << pDenseC.offset( );

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    clsparseIdx_t predicted = subwave_size * pSparseCsrA.num_rows;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    clsparseIdx_t global_work_size =
        group_size* ( ( predicted + group_size - 1 ) / group_size );
    cl::NDRange local( group_size );
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

    cl_int status = kWrapper.run( control, global, local );

    if( status != CL_SUCCESS )
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif
