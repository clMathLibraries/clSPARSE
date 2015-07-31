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
    + " -DROWBITS=" + std::to_string( ROW_BITS )
    + " -DWGBITS=" + std::to_string( WG_BITS )
    + " -DBLOCKSIZE=" + std::to_string( BLKSIZE );
#ifdef DOUBLE
    buildFlags += " -DDOUBLE";
#endif

    if(typeid(T) == typeid(cl_double))
    {
            std::string options = std::string() + " -DDOUBLE";
            params.append(options);
    }


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
    + " -DROWBITS=" + std::to_string( ROW_BITS )
    + " -DWGBITS=" + std::to_string( WG_BITS )
    + " -DBLOCKSIZE=" + std::to_string( BLKSIZE );

    if(typeid(T) == typeid(cl_double))
    {
        std::string options = std::string() + " -DDOUBLE";
        params.append(options);
    }

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
