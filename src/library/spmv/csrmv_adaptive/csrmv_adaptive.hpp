#pragma once
#ifndef _CLSPARSE_CSRMV_ADAPTIVE_HPP_
#define _CLSPARSE_CSRMV_ADAPTIVE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

clsparseStatus
csrmv( const clsparseScalarPrivate& pAlpha, const clsparseCsrMatrixPrivate& pCsrMatx,
            const clsparseVectorPrivate& pX,
            const clsparseScalarPrivate& pBeta,
            clsparseVectorPrivate& pY,
            const std::string& params,
            const cl_uint group_size,
            clsparseControl control )
{
    cl::Kernel kernel = KernelCache::get( control->queue,
                                          "csrmv_adaptive",
                                          "csrmv_adaptive",
                                          params );

    KernelWrap kWrapper( kernel );

    kWrapper << pCsrMatx.values 
        << pCsrMatx.colIndices << pCsrMatx.rowOffsets
        << pX.values << pY.values 
        << pCsrMatx.rowBlocks 
        << pAlpha.value << pBeta.value;
        //<< h_alpha << h_beta;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size = ( pCsrMatx.rowBlockSize - 1 ) * group_size;
    cl::NDRange local( group_size );
    cl::NDRange global( global_work_size > local[ 0 ] ? global_work_size : local[ 0 ] );

    cl_int status = kWrapper.run( control, global, local );

    if( status != CL_SUCCESS )
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

clsparseStatus
clsparseScsrmv_adaptive( const clsparseScalarPrivate& alpha,
        const clsparseCsrMatrixPrivate& pCsrMatx,
        const clsparseVectorPrivate& pX,
        const clsparseScalarPrivate& beta,
        clsparseVectorPrivate& pY,
        clsparseControl control )
{
    clsparseStatus status;

    cl_uint nnz_per_row = pCsrMatx.nnz_per_row( );   //average nnz per row
    const cl_uint wave_size = control->wavefront_size;
    const cl_uint group_size = 256;   //wave_size * 8;    // 256 gives best performance!

    const std::string params = std::string( ) //std::string( "-x clc++ -Dcl_khr_int64_base_atomics=1" )
    + " -DROWBITS=" + std::to_string( ROW_BITS )
    + " -DWGBITS=" + std::to_string( WG_BITS )
    + " -DBLOCKSIZE=" + std::to_string( BLKSIZE );
#ifdef DOUBLE
    buildFlags += " -DDOUBLE";
#endif

    //cl_float h_alpha;
    //cl_float h_beta;

    //{
    //    clMapMemRIAA< cl_float > rAlpha( control->queue( ), alpha.value );
    //    clMapMemRIAA< cl_float > rBeta( control->queue( ), beta.value );
    //    cl_float* pAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, alpha.offset( ), 1 );
    //    cl_float* pBeta = rBeta.clMapMem( CL_TRUE, CL_MAP_READ, beta.offset( ), 1 );
    //    h_alpha = *pAlpha;
    //    h_beta = *pBeta;
    //    std::cout << "h_alpha = " << h_alpha << " h_beta = " << h_beta << std::endl;
    //}

    //y = alpha * A * x + beta * y;
    return csrmv( alpha, pCsrMatx, pX, beta, pY,
                    params, group_size, control );

}

#endif //_CLSPARSE_CSRMV_ADAPTIVE_HPP_
