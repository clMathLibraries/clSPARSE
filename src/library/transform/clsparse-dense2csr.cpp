#include "clSPARSE.h"
#include "internal/clsparse-internal.hpp"
#include "internal/clsparse-validate.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "transform/transform-kernels.h"

//#include <clBLAS.h>
#define HERE printf("HERE\n");fflush(stdout);
// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE-2x.hpp"
#else
    #include "include/clSPARSE-1x.hpp"
#endif

clsparseStatus
clsparseSdense2csr(clsparseCsrMatrix* csr,
                   const cldenseMatrix* A,
                   const clsparseControl control)
{
    clsparseCsrMatrixPrivate* pCsr = static_cast<clsparseCsrMatrixPrivate*>(csr);
    const cldenseMatrixPrivate* pA = static_cast<const cldenseMatrixPrivate*>(A);

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
    cl_int run_status;

    pCsr->num_rows = pA->num_rows;
    pCsr->num_cols = pA->num_cols;

    int total = pA->num_rows * pA->num_cols;
    cl::Context cxt = control->getContext();

    cl_mem scan_input = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           total * sizeof( cl_int ), NULL, &run_status );

    cl_mem scan_output = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           total * sizeof( cl_int ), NULL, &run_status );


    const std::string params = std::string() +
            "-DINDEX_TYPE="    + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DSIZE_TYPE="  + OclTypeTraits<cl_ulong>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"dense2csr", "process_scaninput", params);

    KernelWrap kWrapper(kernel);

    kWrapper << total
             << pA->values
	     << scan_input;

    cl_uint workgroup_size   = 	256;
    cl_uint global_work_size = (total % workgroup_size == 0)? total :  total / workgroup_size * workgroup_size + workgroup_size;
    if (total < workgroup_size) global_work_size = workgroup_size;

    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_work_size);

    run_status = kWrapper.run(control, global, local);

    if (run_status != CL_SUCCESS)
   {
        return clsparseInvalidKernelExecution;
    }

#if 0
    //temporarily on CPU
    int *sum_temp =  (int *)malloc(total * sizeof(int));
    memset(sum_temp, 0, total * sizeof(int));
    run_status = clEnqueueReadBuffer(control->queue(),
                                     scan_input,
                                     1,
                                     0,
                                     total  * sizeof(int),
                                     sum_temp,
                                     0,
                                     0,
                                     0);

    if(run_status != CL_SUCCESS) { fprintf(stderr, "ERROR: read %d\n", run_status);}
    //TODO: temporarily on GPU
    int nnz = 0;
    for(int i = 0; i < total; i++){
        nnz += sum_temp[i];
        //printf("%d ", sum_temp[i]);
    }
    printf("nnz............nnz.........nnz = %d\n", nnz);
    //end on CPU
#endif
    clsparseScalar sum;
    clsparseInitScalar(&sum);

    cldenseVector gY;
    clsparseInitVector(&gY);

    gY.values = clCreateBuffer(cxt(),
                               CL_MEM_READ_WRITE,
                               total * sizeof(cl_int), NULL, &run_status);

    run_status =   clEnqueueCopyBuffer(control->queue(),
                                       scan_input,
                                       gY.values,
                                       0,
                                       0,
                                       total * sizeof(cl_int),
                                       0, NULL, NULL);

    gY.num_values      = total;
    sum.value = clCreateBuffer(cxt(), CL_MEM_READ_WRITE,
                               sizeof(cl_int), NULL, &run_status);

    run_status = cldenseIreduce(&sum, &gY, control);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    int nnz = 0;
    run_status = clEnqueueReadBuffer(control->queue(),
                                     sum.value,
                                     1,
                                     0,
                                     sizeof(cl_int),
                                     &nnz,
                                     0,
                                     NULL,
                                     NULL);


    //printf("nnz = %d\n", nnz);

    status = scan(
                  0,
                  total - 1,
                  scan_input,
                  scan_output,
                  0,
                  1,
                  control
                  );

    if(status != clsparseSuccess)
        return status;

    clsparseCooMatrix cooMatx;
    clsparseInitCooMatrix( &cooMatx );

    cooMatx.num_nonzeros = nnz;
    cooMatx.num_rows = pCsr->num_rows;
    cooMatx.num_cols = pCsr->num_cols;
    //printf("num_rows = %d, num_cols = %d\n", cooMatx.num_rows,  cooMatx.num_cols);

    cooMatx.values     = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_float ), NULL, &run_status );
    cooMatx.colIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &run_status );
    cooMatx.rowIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &run_status );

    cl::Kernel kernel1 = KernelCache::get(control->queue,"dense2csr", "spread_value", params);

    KernelWrap kWrapper1(kernel1);

    kWrapper1 << cooMatx.num_rows << cooMatx.num_cols << total
              << pA->values << scan_input
	      << scan_output
	      << cooMatx.rowIndices
	      << cooMatx.colIndices
	      << cooMatx.values;

    run_status = kWrapper1.run(control, global, local);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    status = clsparseScoo2csr(&cooMatx,
                               csr,
                               control);

    if(status != clsparseSuccess)
        return status;

    clReleaseMemObject(cooMatx.values);
    clReleaseMemObject(cooMatx.colIndices);
    clReleaseMemObject(cooMatx.rowIndices);
    clReleaseMemObject(gY.values);
    clReleaseMemObject(sum.value);

    return clsparseSuccess;

}
clsparseStatus
clsparseDdense2csr(clsparseCsrMatrix* csr,
                   const cldenseMatrix* A,
                   const clsparseControl control)
{
    clsparseCsrMatrixPrivate* pCsr = static_cast<clsparseCsrMatrixPrivate*>(csr);
    const cldenseMatrixPrivate* pA = static_cast<const cldenseMatrixPrivate*>(A);

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
    cl_int run_status;

    pCsr->num_rows = pA->num_rows;
    pCsr->num_cols = pA->num_cols;

    int total = pA->num_rows * pA->num_cols;
    cl::Context cxt = control->getContext();

    cl_mem scan_input = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           total * sizeof( cl_int ), NULL, &run_status );

    cl_mem scan_output = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           total * sizeof( cl_int ), NULL, &run_status );


    const std::string params = std::string() +
            "-DINDEX_TYPE="    + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_double>::type
            + " -DSIZE_TYPE="  + OclTypeTraits<cl_ulong>::type;

    cl::Kernel kernel = KernelCache::get(control->queue,"dense2csr", "process_scaninput", params);

    KernelWrap kWrapper(kernel);

    kWrapper << total
             << pA->values
             << scan_input;

    cl_uint workgroup_size   =  256;
    cl_uint global_work_size = (total % workgroup_size == 0)? total :  total / workgroup_size * workgroup_size + workgroup_size;
    if (total < workgroup_size) global_work_size = workgroup_size;

    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_work_size);

    run_status = kWrapper.run(control, global, local);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    clsparseScalar sum;
    clsparseInitScalar(&sum);

    cldenseVector gY;
    clsparseInitVector(&gY);

    gY.values = clCreateBuffer(cxt(),
                               CL_MEM_READ_WRITE,
                               total * sizeof(cl_int), NULL, &run_status);

    run_status =   clEnqueueCopyBuffer(control->queue(),
                                       scan_input,
                                       gY.values,
                                       0,
                                       0,
                                       total * sizeof(cl_int),
                                       0, NULL, NULL);

    gY.num_values      = total;
    sum.value = clCreateBuffer(cxt(), CL_MEM_READ_WRITE,
                               sizeof(cl_int), NULL, &run_status);

    run_status = cldenseIreduce(&sum, &gY, control);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    int nnz = 0;
    run_status = clEnqueueReadBuffer(control->queue(),
                                     sum.value,
                                     1,
                                     0,
                                     sizeof(cl_int),
                                     &nnz,
                                     0,
                                     NULL,
                                     NULL);


    printf("nnz = %d\n", nnz);

    status = scan(
                  0,
                  total - 1,
                  scan_input,
                  scan_output,
                  0,
                  1,
                  control
                  );

    if(status != clsparseSuccess)
        return status;

    clsparseCooMatrix cooMatx;
    clsparseInitCooMatrix( &cooMatx );

    cooMatx.num_nonzeros = nnz;
    cooMatx.num_rows = pCsr->num_rows;
    cooMatx.num_cols = pCsr->num_cols;
    //printf("num_rows = %d, num_cols = %d\n", cooMatx.num_rows,  cooMatx.num_cols);

    cooMatx.values     = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_double ), NULL, &run_status );
    cooMatx.colIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &run_status );
    cooMatx.rowIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &run_status );

    cl::Kernel kernel1 = KernelCache::get(control->queue,"dense2csr", "spread_value", params);

    KernelWrap kWrapper1(kernel1);

    kWrapper1 << cooMatx.num_rows << cooMatx.num_cols << total
              << pA->values << scan_input
              << scan_output
              << cooMatx.rowIndices
              << cooMatx.colIndices
              << cooMatx.values;

    run_status = kWrapper1.run(control, global, local);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    status = clsparseDcoo2csr(&cooMatx,
                               csr,
                               control);

    if(status != clsparseSuccess)
        return status;

    clReleaseMemObject(cooMatx.values);
    clReleaseMemObject(cooMatx.colIndices);
    clReleaseMemObject(cooMatx.rowIndices);
    clReleaseMemObject(gY.values);
    clReleaseMemObject(sum.value);

    return clsparseSuccess;

}

