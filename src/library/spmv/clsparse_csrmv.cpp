#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include <clBLAS.h>

clsparseStatus
csrmv_a1b0(const int m,
           cl_mem row_offsets, cl_mem col_indices, cl_mem values,
           cl_mem x, cl_mem y,
           const std::string& params,
           const cl_uint group_size,
           const cl_uint subwave_size,
           clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_alpha1_beta0",
                                         "csrmv_alpha1_beta0",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << m
             << row_offsets << col_indices << values
             << x << control->off_x
             << y << control->off_y;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

//    cl::NDRange local(group_size);
//    cl::NDRange global(predicted > local[0] ? predicted : local[0]);
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
csrmv_b0(const int m, cl_mem alpha,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x,
         cl_mem y,
         const std::string& params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_beta0",
                                         "csrmv_beta0",
                                         params);
    KernelWrap kWrapper(kernel);

    kWrapper << m
             << alpha << control->off_alpha
             << row_offsets << col_indices << values
             << x << control->off_x
             << y << control->off_y;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

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
csrmv_a1(const int m,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x,
         cl_mem beta,
         cl_mem y,
         const std::string& params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_alpha1",
                                         "csrmv_alpha1",
                                         params);
    KernelWrap kWrapper(kernel);

    kWrapper << m
             << row_offsets << col_indices << values
             << x << control->off_x
             << beta << control->off_beta
             << y << control->off_y;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

//    cl::NDRange local(group_size);
//    cl::NDRange global(predicted > local[0] ? predicted : local[0]);
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
csrmv_b1(const int m,
         cl_mem alpha,
         cl_mem row_offsets, cl_mem col_indices, cl_mem values,
         cl_mem x,
         cl_mem y,
         const std::string& params,
         const cl_uint group_size,
         const cl_uint subwave_size,
         clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,
                                         "csrmv_beta1",
                                         "csrmv_beta1",
                                         params);

    KernelWrap kWrapper(kernel);

    kWrapper << m
             << alpha << control->off_alpha
             << row_offsets << col_indices << values
             << x << control->off_x
             << y << control->off_y;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

//    cl::NDRange local(group_size);
//    cl::NDRange global(predicted > local[0] ? predicted : local[0]);
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
csrmv(const int m,
      cl_mem alpha,
      cl_mem row_offsets, cl_mem col_indices, cl_mem values,
      cl_mem x,
      cl_mem beta,
      cl_mem y,
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

    kWrapper << m
             << alpha << control->off_alpha
             << row_offsets << col_indices << values
             << x << control->off_x
             << beta << control->off_beta
             << y << control->off_y;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * m;

//    cl::NDRange local(group_size);
//    cl::NDRange global(predicted > local[0] ? predicted : local[0]);
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
clsparseScsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
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

    //validate cl_mem objects
    status = validateMemObject(x, sizeof(cl_float)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_float)*m);
    if(status != clsparseSuccess)
        return status;
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

    status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
    if(status != clsparseSuccess) {
        return status;
    }

    cl_uint nnz_per_row = nnz / m; //average nnz per row
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
    void* h_alpha = clEnqueueMapBuffer(control->queue(), alpha, true, CL_MAP_READ,
                                       0, sizeof(float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), alpha, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_alpha = %f\n", *(float*)h_alpha);
#endif
    void* h_beta = clEnqueueMapBuffer(control->queue(), beta, true, CL_MAP_READ,
                                      0, sizeof(float), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), beta, h_beta, 0, NULL, NULL);

#ifndef NDEBUG
    printf("host_beta = %f\n", *(float*)h_beta);
#endif

    // this functionallity can be implemented in one kernel by using ifdefs
    // passed in parmeters but this way i found more clear;
    if(*(float*)h_alpha == 1.0f && *(float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1, beta = 0\n\n");
#endif
        //y = A*x
        return csrmv_a1b0(m, row_offsets, col_indices, values,
                          x, y,
                          params,
                          group_size,
                          subwave_size,
                          control);
    }
    else if( *(float*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, (clBlasSscale)\n\n");
#endif
        // y = b*y;
        clblasStatus clbls_status =
                clblasSscal(m, *(cl_float*)h_beta, y, control->off_y, 1, 1,
                            &control->queue(),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( ));


        if(clbls_status != clblasSuccess)
            return clsparseInvalidKernelExecution;
        else
            return clsparseSuccess;

    }

    else if( *(float*)h_beta == 0.0)
    {
#ifndef NDEBUG
        printf("\n\nalpha =/= 0, beta = 0\n\n");
#endif
        //y = alpha * A * x
        return csrmv_b0(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x, y,
                        params, group_size, subwave_size,
                        control);
    }

    else if ( *(float*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(m,
                        row_offsets, col_indices, values,
                        x,
                        beta,
                        y,
                        params, group_size, subwave_size,
                        control);
    }

    else if ( *(float*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x,
                        y,
                        params, group_size, subwave_size,
                        control);
    }

    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(m,
                     alpha,
                     row_offsets,
                     col_indices,
                     values,
                     x,
                     beta,
                     y,
                     params, group_size, subwave_size,
                     control);
    }

    return clsparseNotImplemented;
}


//Dummy implementation of new interface;
clsparseStatus
clsparseScsrmv_adaptive( clsparseScalar* alpha,
            clsparseCsrMatrix* matx,
            clsparseVector* x,
            clsparseScalar* beta,
            clsparseVector* y,
            clsparseControl control )
{
    if( matx->rowBlocks == nullptr )
    {
        // Call non-adaptive CSR kernels
        return clsparseScsrmv( matx->m, matx->n, matx->nnz,
                               alpha->value,
                               matx->rowOffsets, matx->colIndices, matx->values,
                               x->values,
                               beta->value,
                               y->values,
                               control );
    }
    else
    {
        // Call adaptive CSR kernels
        return clsparseSuccess;
    }

    return clsparseNotImplemented;
}

clsparseStatus
clsparseDcsrmv(const int m, const int n, const int nnz,
               cl_mem alpha,
               cl_mem row_offsets, cl_mem col_indices, cl_mem values,
               cl_mem x,
               cl_mem beta,
               cl_mem y,
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

    status = validateMemObject(x, sizeof(cl_double)*n);
    if(status != clsparseSuccess)
        return status;
    status = validateMemObject(y, sizeof(cl_double)*m);
    if(status != clsparseSuccess)
        return status;
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

    status = validateMemObjectSize(sizeof(cl_float), 1, alpha, control->off_alpha);
    if(status != clsparseSuccess) {
        return status;
    }

    status = validateMemObjectSize(sizeof(cl_float), 1, beta, control->off_beta);
    if(status != clsparseSuccess) {
        return status;
    }

    cl_uint nnz_per_row = nnz / m; //average nnz per row
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


    //check alpha and beta to distinct kernel version
    void* h_alpha = clEnqueueMapBuffer(control->queue(), alpha, true, CL_MAP_READ,
                                       0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), alpha, h_alpha, 0, NULL, NULL);

#ifndef NDEBUG
    printf("halpha = %g\n", *(cl_double*)h_alpha);
#endif
    void* h_beta = clEnqueueMapBuffer(control->queue(), beta, true, CL_MAP_READ,
                                      0, sizeof(cl_double), 0, NULL, NULL, NULL);
    clEnqueueUnmapMemObject(control->queue(), beta, h_beta, 0, NULL, NULL);

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
        return csrmv_a1b0(m, row_offsets, col_indices, values,
                          x, y,
                          params,
                          group_size,
                          subwave_size,
                          control);
    }
    else if( *(cl_double*)h_alpha == 0.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 0, (clBlasDscale)\n\n");
#endif
        clblasStatus clbls_status =
                clblasDscal(m, *(cl_double*)h_beta, y, control->off_y, 1, 1,
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
        return csrmv_b0(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x, y,
                        params, group_size, subwave_size,
                        control);
    }

    else if ( *(cl_double*)h_alpha == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha = 1.0, beta =/= 0.0\n\n");
#endif
        //y = A*x + b*y
        return csrmv_a1(m,
                        row_offsets, col_indices, values,
                        x,
                        beta,
                        y,
                        params, group_size, subwave_size,
                        control);
    }

    else if ( *(cl_double*)h_beta == 1.0)
    {
#ifndef NDEBUG
        printf("\n\talpha =/= 0, beta = 1.0\n\n");
#endif
        //y = alpha * A * x + y;
        return csrmv_b1(m,
                        alpha,
                        row_offsets, col_indices, values,
                        x,
                        y,
                        params, group_size, subwave_size,
                        control);
    }
    else {
#ifndef NDEBUG
        printf("\n\talpha =/= 0.0, 1.0, beta =/= 0.0, 1.0\n\n");
#endif
        //y = alpha * A * x + beta * y;
        return csrmv(m,
                     alpha,
                     row_offsets,
                     col_indices,
                     values,
                     x,
                     beta,
                     y,
                     params, group_size, subwave_size,
                     control);
    }

    return clsparseNotImplemented;
}
