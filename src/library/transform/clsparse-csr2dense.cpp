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
#include "transform/conversion-utils.hpp"


clsparseStatus
clsparseScsr2dense(const clsparseCsrMatrix* csr,
                    cldenseMatrix* A,
                  const clsparseControl control)
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

    //BUG: For big matrices we might have overflow here;
    size_t dense_size = csr->num_cols * csr->num_rows;

    status = validateMemObject(A->values, dense_size * sizeof(cl_float));

    if(status != clsparseSuccess)
        return status;


    clsparse::vector<size_t>   offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<size_t>   indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_float> values  (control, csr->values,     csr->num_nonzeros);

    clsparse::vector<cl_float> Avalues (control, A->values, dense_size);

    cl_int cl_status = Avalues.fill(control, 0);

    CLSPARSE_V(cl_status, "Error: Fill A values");
    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;


    return transform_csr_2_dense(offsets, indices, values,
                                 csr->num_rows, csr->num_cols, Avalues, control);
}

clsparseStatus
clsparseDcsr2dense(const clsparseCsrMatrix* csr,
cldenseMatrix* A,
                   const clsparseControl control)
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

    //BUG: For big matrices we might have overflow here;
    size_t dense_size = csr->num_cols * csr->num_rows;

    status = validateMemObject(A->values, dense_size * sizeof(cl_double));

    if(status != clsparseSuccess)
        return status;


    clsparse::vector<size_t>   offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<size_t>   indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_double> values  (control, csr->values,     csr->num_nonzeros);

    clsparse::vector<cl_double> Avalues (control, A->values, dense_size);

    cl_int cl_status = Avalues.fill(control, 0);

    CLSPARSE_V(cl_status, "Error: Fill A values");
    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;


    return transform_csr_2_dense(offsets, indices, values,
                                 csr->num_rows, csr->num_cols,Avalues, control);

}
