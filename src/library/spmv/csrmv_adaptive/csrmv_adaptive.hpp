#pragma once
#ifndef _CLSPARSE_CSRMV_ADAPTIVE_HPP_
#define _CLSPARSE_CSRMV_ADAPTIVE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/data_types/clarray.hpp"

template <typename T>
clsparseStatus
csrmv_adaptive( const clsparseScalarPrivate* pAlpha,
                const clsparseCsrMatrixPrivate* pCsrMatx,
                const clsparseVectorPrivate* pX,
                const clsparseScalarPrivate* pBeta,
                clsparseVectorPrivate* pY,
                clsparseControl control )
{
    if(typeid(T) == typeid(cl_double))
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
    cl_uint global_work_size = ( pCsrMatx->rowBlockSize - 1 ) * group_size;
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
csrmv_adaptive( const clsparse::array<T>& pAlpha,
                const clsparseCsrMatrixPrivate* pCsrMatx,
            const clsparse::array<T>& pX,
            const clsparse::array<T>& pBeta,
            clsparse::array<T>& pY,
            clsparseControl control )
{
    if(typeid(T) == typeid(cl_double))
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
                                          "csrmv_adaptive",
                                          "csrmv_adaptive",
                                          params );

    KernelWrap kWrapper( kernel );

    kWrapper << pCsrMatx->values
        << pCsrMatx->colIndices << pCsrMatx->rowOffsets
        << pX.buffer() << pY.buffer()
        << pCsrMatx->rowBlocks
        << pAlpha.buffer() << pBeta.buffer();
        //<< h_alpha << h_beta;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size = ( pCsrMatx->rowBlockSize - 1 ) * group_size;
    cl::NDRange local( group_size );
    cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

    cl_int status = kWrapper.run( control, global, local );

    if( status != CL_SUCCESS )
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

//clsparseStatus
//clsparseScsrmv_adaptive( const clsparseScalarPrivate& alpha,
//        const clsparseCsrMatrixPrivate& pCsrMatx,
//        const clsparseVectorPrivate& pX,
//        const clsparseScalarPrivate& beta,
//        clsparseVectorPrivate& pY,
//        clsparseControl control )
//{


//    //y = alpha * A * x + beta * y;
//    return csrmv_adaptive( alpha, pCsrMatx, pX, beta, pY,
//                    params, group_size, control );

//}

#endif //_CLSPARSE_CSRMV_ADAPTIVE_HPP_
