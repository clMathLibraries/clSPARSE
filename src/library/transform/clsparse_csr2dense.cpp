#include "clSPARSE.h"
#include "internal/clsparse_internal.hpp"
#include "internal/clsparse_validate.hpp"
#include "internal/clsparse_control.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION >= 200 )
    #include "include/clSPARSE-2x.hpp"
#else
    #include "include/clSPARSE-1x.hpp"
#endif


clsparseStatus
csr2dense_transform(const clsparseCsrMatrixPrivate* pCsr,
                    cldenseMatrixPrivate* pA,
                    const std::string& params,
                    const cl_uint group_size,
                    const cl_uint subwave_size,
                    clsparseControl control)
{
    cl::Kernel kernel = KernelCache::get(control->queue,"csr2dense", "csr2dense", params);

    KernelWrap kWrapper(kernel);

    kWrapper << pCsr->m << pCsr->n
             << pCsr->rowOffsets << pCsr->colIndices << pCsr->values
             << pA->values;

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * pCsr->m;

    //cl::NDRange local(group_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);

    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);
    cl::NDRange local(group_size);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    pA->num_rows = pCsr->m;
    pA->num_cols = pCsr->n;

    return clsparseSuccess;
 
}

clsparseStatus
clsparseScsr2dense(const clsparseCsrMatrix* csr,
                    cldenseMatrix* A,
                  const clsparseControl control)
{
    const clsparseCsrMatrixPrivate* pCsr = static_cast<const clsparseCsrMatrixPrivate*>(csr);
    cldenseMatrixPrivate* pA = static_cast<cldenseMatrixPrivate*>( A );

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
    status = validateMemObject(pA->values, sizeof(cl_float)*pCsr->n*pCsr->m);
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    //TODO: ask about validateMemObjectSize
    cl_uint nnz_per_row = pCsr->nnz_per_row(); //average nnz per row
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

    //fill the buffer A with zeros
    cl_float pattern = 0.0f; 
#if (BUILD_CLVERSION >= 200)
    clEnqueueSVMMemFill(control->queue(), pA->values, &pattern, sizeof(cl_float),
                        sizeof(cl_float) * pCsr->m * pCsr->n, 0, NULL, NULL);
#else
    clEnqueueFillBuffer(control->queue(), pA->values, &pattern, sizeof(cl_float), 0,
                        sizeof(cl_float) * pCsr->m * pCsr->n, 0, NULL, NULL);
#endif

    return csr2dense_transform(pCsr,
                               pA,
                               params,
                               group_size,
                               subwave_size,
                               control);

}

clsparseStatus
clsparseDcsr2dense(const clsparseCsrMatrix* csr,
cldenseMatrix* A,
                   const clsparseControl control)
{
    const clsparseCsrMatrixPrivate* pCsr = static_cast<const clsparseCsrMatrixPrivate*>(csr);
    cldenseMatrixPrivate* pA = static_cast<cldenseMatrixPrivate*>( A );

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
    status = validateMemObject(pA->values, sizeof(cl_double)*pCsr->n*pCsr->m);
    if(status != clsparseSuccess)
        return status;

    //validate cl_mem sizes
    //TODO: ask about validateMemObjectSize

    cl_uint nnz_per_row = pCsr->nnz_per_row(); //average nnz per row
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

    //fill the buffer A with zeros
    cl_double pattern = 0.0f;

#if (BUILD_CLVERSION >= 200)
    clEnqueueSVMMemFill(control->queue(), pA->values, &pattern, sizeof(cl_double),
                        sizeof(cl_double) * pCsr->m * pCsr->n, 0, NULL, NULL);
#else
    clEnqueueFillBuffer(control->queue(), pA->values, &pattern, sizeof(cl_double), 0,
                        sizeof(cl_double) * pCsr->m * pCsr->n, 0, NULL, NULL);
#endif
    return csr2dense_transform(pCsr,
                               pA,
                               params,
                               group_size,
                               subwave_size,
                               control);

}

