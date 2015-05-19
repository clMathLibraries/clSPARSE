#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_HPP_
#define _CLSPARSE_CSRMV_VECTOR_HPP_

#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_internal.hpp"

#include "csrmv_vector_impl.hpp"

clsparseStatus
clsparseScsrmv_vector (const clsparseScalarPrivate* pAlpha,
                       const clsparseCsrMatrixPrivate* pMatx,
                       const clsparseVectorPrivate* pX,
                       const clsparseScalarPrivate* pBeta,
                       clsparseVectorPrivate* pY,
                       clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseStatus status;

    /*TODO: make it work for CL1.x and CL2.x
     */
#if defined(_DEBUG)

    ////validate cl_mem objects
    //status = validateMemObject(x, sizeof(cl_float)*n);
    //if(status != clsparseSuccess)
    //    return status;
    //status = validateMemObject(y, sizeof(cl_float)*m);
    //if(status != clsparseSuccess)
    //    return status;

    ///*
    // * TODO: take care of the offsets !!!
    // */
    //status = validateMemObject(alpha, sizeof(cl_float));
    //if(status != clsparseSuccess)
    //    return status;
    //status = validateMemObject(beta, sizeof(cl_float));
    //if(status != clsparseSuccess)
    //    return status;

    ////validate cl_mem sizes
    //status = validateMemObjectSize(sizeof(cl_float), n, x, control->off_x);
    //if(status != clsparseSuccess) {
    //    return status;
    //}

    //status = validateMemObjectSize(sizeof(cl_float), m, y, control->off_y);
    //if(status != clsparseSuccess) {
    //    return status;
    //}

    ///*
    // * TODO: take care of the offsets !!!
    // */
    //status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
    //if(status != clsparseSuccess) {
    //    return status;
    //}

    //status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
    //if(status != clsparseSuccess) {
    //    return status;
    //}
#endif



    return csrmv<cl_float>(pAlpha, pMatx, pX, pBeta, pY, control);



}


clsparseStatus
clsparseDcsrmv_vector(const clsparseScalarPrivate* pAlpha,
                      const clsparseCsrMatrixPrivate* pMatx,
                      const clsparseVectorPrivate* pX,
                      const clsparseScalarPrivate* pBeta,
                      clsparseVectorPrivate* pY,
                      clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseStatus status;

    /*TODO: make it work for CL1.x and CL2.x
     */
//#if defined(_DEBUG)
//    status = validateMemObject(x, sizeof(cl_double)*n);
//    if(status != clsparseSuccess)
//        return status;
//    status = validateMemObject(y, sizeof(cl_double)*m);
//    if(status != clsparseSuccess)
//        return status;
//    /*
//     * TODO: take care of the offsets !!!
//     */
//    status = validateMemObject(alpha, sizeof(cl_double));
//    if(status != clsparseSuccess)
//        return status;
//    status = validateMemObject(beta, sizeof(cl_double));
//    if(status != clsparseSuccess)
//        return status;
//
//    //validate cl_mem sizes
//    status = validateMemObjectSize(sizeof(cl_float), n, x, control->off_x);
//    if(status != clsparseSuccess) {
//        return status;
//    }
//
//    status = validateMemObjectSize(sizeof(cl_float), m, y, control->off_y);
//    if(status != clsparseSuccess) {
//        return status;
//    }
//
//    /*
//     * TODO: take care of the offsets !!!
//     */
//    status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
//    if(status != clsparseSuccess) {
//        return status;
//    }
//
//    status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
//    if(status != clsparseSuccess) {
//        return status;
//    }
//#endif


    //y = alpha * A * x + beta * y;
     return csrmv<cl_double>(pAlpha, pMatx, pX, pBeta, pY, control);

}

#endif //_CLSPARSE_CSRMV_VECTOR_HPP_
