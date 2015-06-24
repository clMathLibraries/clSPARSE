#include "clSPARSE.h"
#include "internal/clsparse-internal.hpp"
#include "internal/clsparse-validate.hpp"
#include "internal/clsparse-control.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"
#include "transform/transform-kernels.h"

//#include <clBLAS.h>

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE-2x.hpp"
#else
    #include "include/clSPARSE-1x.hpp"
#endif

#define HERE printf("here\n");fflush(stdout);

//COO <--> CSR
clsparseStatus
clsparse_coo2csr_internal(const clsparseCooMatrix* coo,
                          clsparseCsrMatrix* csr,
                          const clsparseControl control,
                          int float_type){

    const clsparseCooMatrixPrivate* pCoo = static_cast<const clsparseCooMatrixPrivate*>(coo);
    clsparseCsrMatrixPrivate* pCsr = static_cast<clsparseCsrMatrixPrivate*>(csr);

    clsparseStatus status;
    cl_int err;

    if (!clsparseInitialized)
    {
       return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
       return clsparseInvalidControlObject;
    }

    cl::Context context = control->getContext();
    pCsr->m = pCoo->m;
    pCsr->n = pCoo->n;
    pCsr->nnz = pCoo->nnz;

    cl_mem rowIndices  = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->nnz)*sizeof(int), NULL, &err );
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);

    clEnqueueCopyBuffer(control->queue(),
                        pCoo-> rowIndices,
                        rowIndices,
                        0,
                        0,
                        sizeof(cl_int) * pCsr->nnz,
                        0,
                        NULL,
                        NULL);


    clEnqueueCopyBuffer(control->queue(),
                        pCoo-> colIndices,
                        pCsr-> colIndices,
                        0,
                        0,
                        sizeof(cl_int) * pCsr->nnz,
                        0,
                        NULL,
                        NULL);

    int f_size = (float_type==0)? sizeof(cl_float):sizeof(cl_double); //todo make a template

    clEnqueueCopyBuffer(control->queue(),
                        pCoo-> values,
                        pCsr-> values,
                        0,
                        0,
                        f_size * pCsr->nnz,
                        0,
                        NULL,
                        NULL);


    status = radix_sort_by_key(
                               0,
                               pCoo->nnz - 1,
                               0,
                               rowIndices,
                               pCsr->colIndices,
                               pCsr->values,
                               float_type,
                               control);

    if(status != clsparseSuccess)
        return status;

    cl_mem one_array  = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->nnz)*sizeof(int), NULL, &err );
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);
    cl_mem row_indices_out  = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->nnz)*sizeof(int), NULL, &err );
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);

    cl_mem scan_input = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->m + 1)*sizeof(int), NULL, NULL);
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);
    //cl_mem scan_output = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->m + 1)*sizeof(int), NULL, NULL);
    //if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);

    int pattern = 1, zero = 0;
    err = clEnqueueFillBuffer(control->queue(), one_array, &pattern, sizeof(int), 0,
                          (pCsr->nnz)*sizeof(int), 0, NULL, NULL);
    if(err != CL_SUCCESS) fprintf(stderr, "ERROR: clFillBuffer  %d\n",  err);
    err = clEnqueueFillBuffer(control->queue(), scan_input, &zero, sizeof(int), 0,
                          (pCsr->m + 1)*sizeof(int), 0, NULL, NULL);
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clFillBuffer  %d\n",  err);

    int count;
    status = reduce_by_key(0,
                           pCoo->nnz -1,
                           0,
                           rowIndices,
                           one_array,
                           row_indices_out,
                           one_array,
		           &count,
		           control);

    if(status != clsparseSuccess)
        return status;

    const std::string params = std::string() + " ";

    cl::Kernel kernel = KernelCache::get(control->queue,"prescan_scatter", "prescan_scatter", params);
    KernelWrap kWrapper(kernel);

    int workgroup_size = 256;
    int global_size    = (count%workgroup_size==0)? count: (count/workgroup_size + 1) * workgroup_size;
    if (count < workgroup_size) global_size = workgroup_size;

    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_size);

    kWrapper  //<< pCoo->rowIndices
             << row_indices_out
             << one_array
             << scan_input
             << count;

    err = kWrapper.run(control, global, local);

    if (err != CL_SUCCESS)
    {
      return clsparseInvalidKernelExecution;
    }

    status = scan( 0,
             pCsr->m,
             scan_input,
             pCsr->rowOffsets,
             0,
             1,
             control);

    if(status != clsparseSuccess)
        return status;

    clReleaseMemObject(one_array);
    clReleaseMemObject(rowIndices);
    clReleaseMemObject(scan_input);
    clReleaseMemObject(row_indices_out);

    return clsparseSuccess;

}

clsparseStatus
clsparseScoo2csr(const clsparseCooMatrix* coo,
                     clsparseCsrMatrix* csr,
                     const clsparseControl control){


   return clsparse_coo2csr_internal(coo,
                                    csr,
                                    control,
                                    0);

}

clsparseStatus
clsparseDcoo2csr(const clsparseCooMatrix* coo,
                     clsparseCsrMatrix* csr,
                     const clsparseControl control){

   return clsparse_coo2csr_internal(coo,
                                    csr,
                                    control,
                                    1);

}
