#pragma once
#ifndef _CLSPARSE_CSRMM_HPP_
#define _CLSPARSE_CSRMM_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

template <typename T>
clsparseStatus
csrmm( const clsparseScalarPrivate& pAlpha,
const clsparseCsrMatrixPrivate& pSparseCsrA,
const cldenseMatrixPrivate& pDenseB,
const clsparseScalarPrivate& pBeta,
cldenseMatrixPrivate& pDenseC,
const clsparseControl control )
{
  if( typeid( T ) == typeid( cl_double ) )
  {
      return clsparseNotImplemented;
  }

  const cl_uint group_size = 256;

  const std::string params = std::string( )
  + " -DROWBITS=" + std::to_string( ROW_BITS )
  + " -DWGBITS=" + std::to_string( WG_BITS )
  + " -DBLOCKSIZE=" + std::to_string( BLKSIZE );
#ifdef DOUBLE
  buildFlags += " -DDOUBLE";
#endif

  cl::Kernel kernel = KernelCache::get( control->queue,
                                        "csrmm",
                                        "csrmm",
                                        params );

  KernelWrap kWrapper( kernel );

  kWrapper << pSparseCsrA.values
      << pSparseCsrA.colIndices << pSparseCsrA.rowOffsets
      << pDenseB.values << pDenseC.values
      << pSparseCsrA.rowBlocks
      << pAlpha.value << pBeta.value;

  // if NVIDIA is used it does not allow to run the group size
  // which is not a multiplication of group_size. Don't know if that
  // have an impact on performance
  cl_uint global_work_size = ( pSparseCsrA.rowBlockSize - 1 ) * group_size;
  cl::NDRange local( group_size );
  cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

  cl_int status = kWrapper.run( control, global, local );

  if( status != CL_SUCCESS )
  {
      return clsparseInvalidKernelExecution;
  }

  return clsparseSuccess;
}

#endif
