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
#include "internal/clsparse-control.hpp"
#include "conversion-utils.hpp"

clsparseStatus
clsparseScoo2csr (const clsparseCooMatrix* coo,
                   clsparseCsrMatrix* csr,
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

    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;

    // how to obtain proper type of the matrix indices? int assumed
    clsparse::vector<clsparseIdx_t> csr_row_offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<clsparseIdx_t> csr_col_indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_float> csr_values (control, csr->values, csr->num_nonzeros);

    clsparse::vector<clsparseIdx_t> coo_row_indices (control, coo->rowIndices, coo->num_nonzeros);
    clsparse::vector<clsparseIdx_t> coo_col_indices (control, coo->colIndices, coo->num_nonzeros);
    clsparse::vector<cl_float> coo_values (control, coo->values, coo->num_nonzeros);

    csr_col_indices = coo_col_indices;
    csr_values = coo_values;

    clsparseStatus status = indices_to_offsets(csr_row_offsets, coo_row_indices, control);
    CLSPARSE_V(status, "Error: coo2csr indices to offsets");

    return status;

}


clsparseStatus
clsparseDcoo2csr ( const clsparseCooMatrix* coo,
                   clsparseCsrMatrix* csr,
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

    csr->num_rows = coo->num_rows;
    csr->num_cols = coo->num_cols;
    csr->num_nonzeros = coo->num_nonzeros;

    // how to obtain proper type of the matrix indices? int assumed
    clsparse::vector<clsparseIdx_t> csr_row_offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<clsparseIdx_t> csr_col_indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_double> csr_values (control, csr->values, csr->num_nonzeros);

    clsparse::vector<clsparseIdx_t> coo_row_indices (control, coo->rowIndices, coo->num_nonzeros);
    clsparse::vector<clsparseIdx_t> coo_col_indices (control, coo->colIndices, coo->num_nonzeros);
    clsparse::vector<cl_double> coo_values (control, coo->values, coo->num_nonzeros);

    csr_col_indices = coo_col_indices;
    csr_values = coo_values;

    clsparseStatus status = indices_to_offsets(csr_row_offsets, coo_row_indices, control);
    CLSPARSE_V(status, "Error: coo2csr indices to offsets");

    return status;

}


