#pragma once
#ifndef _CLSPARSE_PRECOND_UTILS_HPP_
#define _CLSPARSE_PRECOND_UTILS_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"
#include "internal/clsparse_internal.hpp"


template<typename T, bool inverse = false>
clsparseStatus
extract_diagonal(clsparseVectorPrivate* pDiag,
                 const clsparseCsrMatrixPrivate* pA,
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

    assert (pA->m > 0);
    assert (pA->n > 0);
    assert (pA->nnz > 0);

    assert (pDiag->n == min(pA->n, pA->m));

    cl_ulong wg_size = 256;
    cl_ulong size = pA->m;

    cl_ulong nnz_per_row = pA->nnz_per_row();
    cl_ulong wave_size = control->wavefront_size;
    cl_ulong subwave_size = wave_size;

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


    std::string params = std::string()
            + " -DSIZE_TYPE=" + OclTypeTraits<cl_ulong>::type
            + " -DINDEX_TYPE=" + OclTypeTraits<cl_int>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DWG_SIZE=" + std::to_string(wg_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    if (inverse)
        params.append(" -DOP_DIAG_INVERSE");


    cl::Kernel kernel = KernelCache::get(control->queue, "matrix_utils",
                                         "extract_diagonal", params);

    KernelWrap kWrapper(kernel);

    kWrapper << size
             << pDiag->values
             << pA->rowOffsets
             << pA->colIndices
             << pA->values;

    cl_uint predicted = subwave_size * size;

    cl_uint global_work_size =
            wg_size * ((predicted + wg_size - 1 ) / wg_size);
    cl::NDRange local(wg_size);
    //cl::NDRange global(predicted > local[0] ? predicted : local[0]);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int status = kWrapper.run(control, global, local);

    if (status != CL_SUCCESS)
    {
        return clsparseInvalidKernelExecution;
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_PRECOND_UTILS_HPP_
