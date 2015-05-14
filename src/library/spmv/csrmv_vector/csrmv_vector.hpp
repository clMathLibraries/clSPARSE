#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_HPP_
#define _CLSPARSE_CSRMV_VECTOR_HPP_

#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_internal.hpp"

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
#include "include/clSPARSE_2x.hpp"
#else
#include "include/clSPARSE_1x.hpp"
#endif

clsparseStatus
csrmv (const clsparseScalarPrivate* pAlpha,
       const clsparseCsrMatrixPrivate* pMatx,
       const clsparseVectorPrivate* pX,
       const clsparseScalarPrivate* pBeta,
       clsparseVectorPrivate* pY,
       const std::string& params,
       const cl_uint group_size,
       const cl_uint subwave_size,
       clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_general",
                                         "csrmv_general",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << pMatx->m
             << pAlpha->value << pAlpha->offset()
             << pMatx->rowOffsets
             << pMatx->colIndices
             << pMatx->values
             << pX->values << pX->offset()
             << pBeta->value << pBeta->offset()
             << pY->values << pY->offset();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pMatx->m;

    // if NVIDIA is used it does not allow to run the group size
    // which is not a multiplication of group_size. Don't know if that
    // have an impact on performance
    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;

}

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

    cl_uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = 256; //wave_size * 8;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }


    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);


    return csrmv(pAlpha, pMatx, pX, pBeta, pY,
                     params, group_size, subwave_size, control);



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

    cl_uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = 256;    // 256 gives best performance!
    cl_uint subwave_size = wave_size;

    // adjust subwave_size according to nnz_per_row;
    // each wavefron will be assigned to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (nnz_per_row < 64) {  subwave_size = 32;  }
    }
    if (nnz_per_row < 32) {  subwave_size = 16;  }
    if (nnz_per_row < 16) {  subwave_size = 8;  }
    if (nnz_per_row < 8)  {  subwave_size = 4;  }
    if (nnz_per_row < 4)  {  subwave_size = 2;  }

    const std::string params = std::string() +
            "-DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);


    //y = alpha * A * x + beta * y;
     return csrmv(pAlpha, pMatx, pX, pBeta, pY,
                  params, group_size, subwave_size, control);

}

#endif //_CLSPARSE_CSRMV_VECTOR_HPP_
