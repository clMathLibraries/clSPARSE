#pragma once
#ifndef _CLSPARSE_CSRMV_VECTOR_HPP_
#define _CLSPARSE_CSRMV_VECTOR_HPP_

#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_internal.hpp"
#include "spmv/csrmv_vector/csrmv_vector_impl.hpp"
#include <clBLAS.h>

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
#include "include/clSPARSE_2x.hpp"
#else
#include "include/clSPARSE_1x.hpp"
#endif


//clsparseStatus
//clsparseScsrmv_vector(const int m, const int n, const int nnz,
//               cl_mem alpha,
//               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
//               cl_mem x,
//               cl_mem beta,
//               cl_mem y,
//               clsparseControl control)
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

    //validate cl_mem objects
    status = validateMemObject(x, sizeof(cl_float)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_float)*m);
    if(status != clsparseSuccess)
        return status;

    /*
     * TODO: take care of the offsets !!!
     */
    status = validateMemObject(alpha, sizeof(cl_float));
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(beta, sizeof(cl_float));
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    status = validateMemObjectSize(sizeof(cl_float), n, x, control->off_x);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), m, y, control->off_y);
    if(status != clsparseSuccess) {
        return status;
    }

    /*
     * TODO: take care of the offsets !!!
     */
    status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
    if(status != clsparseSuccess) {
        return status;
    }
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


    //check alpha and beta to distinct kernel version
    void* h_alpha = clEnqueueMapBuffer(control->queue(), pAlpha->value, true, CL_MAP_READ,
                                       0, sizeof(cl_float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), pAlpha->value, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_alpha = %f\n", *(cl_float*)h_alpha);
#endif
    //TODO: Cover it with some nice function which can take CL1.x and CL2.x structures
    void* h_beta = clEnqueueMapBuffer(control->queue(), pBeta->value, true, CL_MAP_READ,
                                      0, sizeof(cl_float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), pBeta->value, h_beta, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_beta = %f\n", *(cl_float*)h_beta);
#endif

    // this functionallity can be implemented in one kernel by using ifdefs
    // passed in parmeters but this way i found more clear;
    if(*(float*)h_alpha == 1.0f && *(cl_float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1, beta = 0\n\n");
#endif
        //y = A*x
        return csrmv_a1b0(pMatx, pX, pY,
                          params, group_size, subwave_size, control);
    }
    else if( *(cl_float*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, (clBlasSscale)\n\n");
#endif
        //TODO: write internal scale function which will takes clsparse[]Private
        //      data pointers as arguments. Inside of it call original clblas or
        //      cl2.x implementation
        // y = b*y;
        clblasStatus clbls_status =
                clblasSscal(pMatx->m, *(cl_float*)h_beta, pY->values, pY->offValues,
                            1, 1,
                            &control->queue(),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( ));

        if(clbls_status != clblasSuccess)
            return clsparseInvalidKernelExecution;
        else
            return clsparseSuccess;

    }

    else if( *(cl_float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\nalpha =/= 0, beta = 0\n\n");
#endif
        //y = alpha * A * x
        return csrmv_b0(pAlpha, pMatx, pX, pY,
                        params, group_size, subwave_size, control);
    }

    else if ( *(float*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(pMatx, pX, pBeta, pY,
                        params, group_size, subwave_size, control);
    }

    else if ( *(cl_float*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(pAlpha, pMatx, pX, pY,
                        params, group_size, subwave_size, control);
    }

    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(pAlpha, pMatx, pX, pBeta, pY,
                     params, group_size, subwave_size, control);
    }

    return clsparseNotImplemented;
}


clsparseStatus
clsparseSDsrmv_vector(const clsparseScalarPrivate* pAlpha,
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
    status = validateMemObject(x, sizeof(cl_double)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_double)*m);
    if(status != clsparseSuccess)
        return status;
    /*
     * TODO: take care of the offsets !!!
     */
    status = validateMemObject(alpha, sizeof(cl_double));
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(beta, sizeof(cl_double));
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    status = validateMemObjectSize(sizeof(cl_float), n, x, control->off_x);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), m, y, control->off_y);
    if(status != clsparseSuccess) {
        return status;
    }

    /*
     * TODO: take care of the offsets !!!
     */
    status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
    if(status != clsparseSuccess) {
        return status;
    }
#endif

    cl_uint nnz_per_row = pMatx->nnz_per_row(); //average nnz per row
    cl_uint wave_size = control->wavefront_size;
    cl_uint group_size = wave_size * 4;    // 256 gives best performance!
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


    /*
     * TODO: take care of the offsets for scalars !!!
     */
    //check alpha and beta to distinct kernel version
    void* h_alpha = clEnqueueMapBuffer(control->queue(), pAlpha->value, true, CL_MAP_READ,
                                       0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), pAlpha->value, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("halpha = %g\n", *(cl_double*)h_alpha);
#endif
    void* h_beta = clEnqueueMapBuffer(control->queue(), pBeta->value, true, CL_MAP_READ,
                                      0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), pBeta->value, h_beta, 0, NULL, NULL);

#ifndef NDEBUG
    printf("hbeta= %g\n", *(cl_double*)h_beta);
#endif

    // this functionallity can be implemented in one kernel by using ifdefs
    // passed in parmeters but this way i found more clear;
    if(*(cl_double*)h_alpha == 1.0 && *(cl_double*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1, beta = 0\n\n");
#endif
        //y = A*x
        return csrmv_a1b0(pMatx, pX, pY,
                          params, group_size, subwave_size, control);
    }
    else if( *(cl_double*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, (clBlasDscale)\n\n");
#endif
        clblasStatus clbls_status =
                clblasDscal(pMatx->m, *(cl_double*)h_beta, pY->values,
                            pY->offValues, 1, 1,
                            &control->queue(),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event());

        if (clbls_status != clblasSuccess)
            return clsparseInvalidKernelExecution;
        else
            return clsparseSuccess;
    }

    else if( *(cl_double*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 0\n\n");
#endif
        //y = alpha * A * x;
        return csrmv_b0(pAlpha, pMatx, pX, pY,
                        params, group_size, subwave_size, control);
    }

    else if ( *(cl_double*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(pMatx, pX, pBeta, pY,
                        params, group_size, subwave_size, control);
    }

    else if ( *(cl_double*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(pAlpha, pMatx, pX, pY,
                        params, group_size, subwave_size, control);
    }
    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(pAlpha, pMatx, pX, pBeta, pY,
                     params, group_size, subwave_size, control);
    }

    return clsparseNotImplemented;
}







#endif //_CLSPARSE_CSRMV_VECTOR_HPP_
