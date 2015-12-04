/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-internal.hpp"
#include "internal/clsparse-validate.hpp"
#include "internal/clsparse-control.hpp"

#include "internal/data-types/clvector.hpp"
#include "transform/scan.hpp"
#include "transform/conversion-utils.hpp"

clsparseStatus
clsparseSdense2csr(const cldenseMatrix* A, clsparseCsrMatrix* csr,
                    const clsparseControl control)
{
    typedef cl_float ValueType;
    typedef clsparseIdx_t IndexType;
    typedef cl_ulong SizeType;

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

    SizeType dense_size = A->num_cols * A->num_rows;
    clsparse::vector<ValueType> Avalues (control, A->values, dense_size);

    //calculate nnz
    clsparse::vector<IndexType> nnz_locations (control, dense_size,
                                               0, CL_MEM_READ_WRITE, false);

    IndexType num_nonzeros = 0;

    status = calculate_num_nonzeros(Avalues, nnz_locations, num_nonzeros, control);

    CLSPARSE_V(status, "Error: calculate num nonzeros");
    if (status!= clsparseSuccess)
        return clsparseInvalidKernelExecution;


    clsparse::vector<IndexType> coo_indexes(control, dense_size, 0, CL_MEM_READ_WRITE, false);
    status = exclusive_scan<EW_PLUS>(coo_indexes, nnz_locations, control);

    CLSPARSE_V(status, "Error: exclusive scan");
    if (status!= clsparseSuccess)
        return clsparseInvalidKernelExecution;

    clsparseCooMatrix coo;

    clsparseInitCooMatrix(&coo);

    coo.num_nonzeros = num_nonzeros;
    coo.num_rows = A->num_rows;
    coo.num_cols = A->num_cols;

    //coo is allocated inside this functions
    status = dense_to_coo(&coo, Avalues, nnz_locations, coo_indexes, control);

    CLSPARSE_V(status, "Error: dense_to_coo");
    if (status != clsparseSuccess)
        return clsparseInvalidKernelExecution;

    status = clsparseScoo2csr(&coo, csr, control);


    clReleaseMemObject(coo.values);
    clReleaseMemObject(coo.colIndices);
    clReleaseMemObject(coo.rowIndices);

    return status;
}


clsparseStatus
clsparseDdense2csr(const cldenseMatrix* A,
                   clsparseCsrMatrix* csr,
                   const clsparseControl control)
{
    typedef cl_double ValueType;
    typedef clsparseIdx_t IndexType;
    typedef cl_ulong SizeType;

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

    SizeType dense_size = A->num_cols * A->num_rows;
    clsparse::vector<ValueType> Avalues (control, A->values, dense_size);

    //calculate nnz
    clsparse::vector<IndexType> nnz_locations (control, dense_size,
                                               0, CL_MEM_READ_WRITE, false);

    IndexType num_nonzeros = 0;

    status = calculate_num_nonzeros(Avalues, nnz_locations, num_nonzeros, control);

    CLSPARSE_V(status, "Error: calculate num nonzeros");
    if (status!= clsparseSuccess)
        return clsparseInvalidKernelExecution;


    clsparse::vector<IndexType> coo_indexes(control, dense_size, 0, CL_MEM_READ_WRITE, false);
    status = exclusive_scan<EW_PLUS>(coo_indexes, nnz_locations, control);

    CLSPARSE_V(status, "Error: exclusive scan");
    if (status!= clsparseSuccess)
        return clsparseInvalidKernelExecution;

    clsparseCooMatrix coo;

    clsparseInitCooMatrix(&coo);

    coo.num_nonzeros = num_nonzeros;
    coo.num_rows = A->num_rows;
    coo.num_cols = A->num_cols;

    //coo is allocated inside this functions
    status = dense_to_coo(&coo, Avalues, nnz_locations, coo_indexes, control);

    CLSPARSE_V(status, "Error: dense_to_coo");
    if (status != clsparseSuccess)
        return clsparseInvalidKernelExecution;

    status = clsparseDcoo2csr(&coo, csr, control);


    clReleaseMemObject(coo.values);
    clReleaseMemObject(coo.colIndices);
    clReleaseMemObject(coo.rowIndices);

    return status;

}

