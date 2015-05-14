#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "transform/transform_kernels.h"

#include <clBLAS.h>
#define HERE printf("HERE\n");fflush(stdout);
// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE_2x.hpp"
#else
    #include "include/clSPARSE_1x.hpp"
#endif

clsparseStatus
clsparseSdense2csr(clsparseCsrMatrix* csr,
                   clsparseDenseMatrix* A,
                   const clsparseControl control)
{
    clsparseCsrMatrixPrivate* pCsr = static_cast<clsparseCsrMatrixPrivate*>(csr);
    clsparseDenseMatrixPrivate* pA = static_cast<clsparseDenseMatrixPrivate*>(A);

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
	
    pCsr->m = pA->m;
    pCsr->n = pA->n;
	
    cl::Context cxt = control->getContext();
    cl_mem scan_input = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           A->m * A->n * sizeof( cl_int ), NULL, &run_status );

    cl_mem scan_output = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           A->m * A->n * sizeof( cl_int ), NULL, &run_status );
    

    const std::string params = std::string() +
            "-DINDEX_TYPE="    + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<cl_float>::type
            + " -DSIZE_TYPE="  + OclTypeTraits<cl_ulong>::type;
   			
    cl::Kernel kernel = KernelCache::get(control->queue,"dense2csr", "process_scaninput", params);

    KernelWrap kWrapper(kernel);

    int total = pA->m * pA->n;
    
    kWrapper << total
             << pA->values
	     << scan_input;


    //if(run_status != CL_SUCCESS) { fprintf(stderr, "ERROR: read %d\n", run_status);}
    cl_uint workgroup_size   = 	256;		 
    cl_uint global_work_size = ((pA->m * pA->n) % workgroup_size == 0)? (pA->m * pA->n) :  (pA->m * pA->n) / workgroup_size * workgroup_size + workgroup_size;
    if (pA->m * pA->n < workgroup_size) global_work_size = workgroup_size;
	
    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_work_size);

    run_status = kWrapper.run(control, global, local);

    if (run_status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }
	
    //temporarily on CPU
    int *sum_temp =  (int *)malloc((pA->m * pA->n) * sizeof(int));
    memset(sum_temp, 0, (pA->m * pA->n) * sizeof(int)); 
    run_status = clEnqueueReadBuffer(control->queue(), 
                                     scan_input, 
                                     1, 
                                     0, 
                                    (pA->m *pA->n)  * sizeof(int), 
                                     sum_temp, 
                                     0, 
                                     0, 
                                     0);	

    if(run_status != CL_SUCCESS) { fprintf(stderr, "ERROR: read %d\n", run_status);}
    //temporarily on GPU
    int nnz = 0;
    for(int i = 0; i < pA->m * pA->n; i++)
        nnz += sum_temp[i];
    printf("nnz............nnz.........nnz = %d\n", nnz);
    //end on CPU

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
	
    cooMatx.nnz = nnz;
    cooMatx.m = pCsr->m;
    cooMatx.n = pCsr->n;
	
    cooMatx.values     = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.nnz * sizeof( cl_float ), NULL, &run_status );
    cooMatx.colIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &run_status );
    cooMatx.rowIndices = ::clCreateBuffer( cxt(), CL_MEM_READ_WRITE,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &run_status );

    cl::Kernel kernel1 = KernelCache::get(control->queue,"dense2csr", "spread_value", params);

    KernelWrap kWrapper1(kernel1);

    kWrapper1 << pA->m << pA->n << total
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
	
    status = clsparseScoo2csr_GPU(&cooMatx,
                                   csr,
                                   control);

    if(status != clsparseSuccess)
        return status;

    return clsparseSuccess;

}

