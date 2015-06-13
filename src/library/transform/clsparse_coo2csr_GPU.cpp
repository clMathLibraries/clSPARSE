#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "transform/transform_kernels.h"

//#include <clBLAS.h>

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE_2x.hpp"
#else
    #include "include/clSPARSE_1x.hpp"
#endif

#define HERE printf("here\n");fflush(stdout);

//COO <--> CSR
clsparseStatus
clsparse_coo2csr_internal(clsparseCooMatrix* coo,
                          clsparseCsrMatrix* csr,
                          clsparseControl control,
                          int float_type){
					 
    clsparseCooMatrixPrivate* pCoo = static_cast<clsparseCooMatrixPrivate*>(coo);
    clsparseCsrMatrixPrivate* pCsr = static_cast<clsparseCsrMatrixPrivate*>(csr);
    
    clsparseStatus status;
	
    if (!clsparseInitialized)
    {
       return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
       return clsparseInvalidControlObject;
    }
	
    pCsr->m = pCoo->m;
    pCsr->n = pCoo->n;
    pCsr->nnz = pCoo->nnz;      

    cl::Context context = control->getContext();
    
    status = radix_sort_by_key(
                               0,
                               pCoo->nnz - 1,
                               0,
                               pCoo->rowIndices,
                               pCoo->colIndices,
                               pCoo->values,
                               float_type, 
                               control);

    if(status != clsparseSuccess)
        return status;
        
    cl_int err;				  
    cl_mem one_array  = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->nnz)*sizeof(int), NULL, &err );
    if(err != CL_SUCCESS)  fprintf(stderr, "ERROR: clCreateBuffer  %d\n",  err);
    cl_mem row_indices  = clCreateBuffer(context(), CL_MEM_READ_WRITE, (pCsr->nnz)*sizeof(int), NULL, &err );
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
                           pCoo->rowIndices,
                           one_array,
                           row_indices,
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
             << row_indices
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

    clReleaseMemObject(one_array);
    clReleaseMemObject(row_indices);
    clReleaseMemObject(scan_input);
    //clReleaseMemObject(scan_output); 
     
    return clsparseSuccess;

}

clsparseStatus
clsparseScoo2csr_GPU(clsparseCooMatrix* coo,
                     clsparseCsrMatrix* csr,
                     clsparseControl control){

   
   return clsparse_coo2csr_internal(coo,
                                    csr,
                                    control,
                                    0);

}

clsparseStatus
clsparseDcoo2csr_GPU(clsparseCooMatrix* coo,
                     clsparseCsrMatrix* csr,
                     clsparseControl control){

   return clsparse_coo2csr_internal(coo,
                                    csr,
                                    control,
                                    1);

}

