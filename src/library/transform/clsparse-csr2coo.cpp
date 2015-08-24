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
clsparseScsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
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

    coo->num_rows = csr->num_rows;
    coo->num_cols = csr->num_cols;
    coo->num_nonzeros = csr->num_nonzeros;

    // how to obtain proper type of the matrix indices? int assumed
    clsparse::vector<int> csr_row_offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<int> csr_col_indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_float> csr_values (control, csr->values, csr->num_nonzeros);

    clsparse::vector<int> coo_row_indices (control, coo->rowIndices, coo->num_nonzeros);
    clsparse::vector<int> coo_col_indices (control, coo->colIndices, coo->num_nonzeros);
    clsparse::vector<cl_float> coo_values (control, coo->values, coo->num_nonzeros);

    coo_col_indices = csr_col_indices;
    coo_values = csr_values;


    clsparseStatus status = offsets_to_indices(coo_row_indices, csr_row_offsets, csr->num_rows, control);

    CLSPARSE_V(status, "Error: offsets_to_indices");

    return status;
}

clsparseStatus
clsparseDcsr2coo(const clsparseCsrMatrix* csr,
                 clsparseCooMatrix* coo,
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

    coo->num_rows = csr->num_rows;
    coo->num_cols = csr->num_cols;
    coo->num_nonzeros = csr->num_nonzeros;

    // how to obtain proper type of the matrix indices? int assumed
    clsparse::vector<int> csr_row_offsets (control, csr->rowOffsets, csr->num_rows + 1);
    clsparse::vector<int> csr_col_indices (control, csr->colIndices, csr->num_nonzeros);
    clsparse::vector<cl_double> csr_values (control, csr->values, csr->num_nonzeros);

    clsparse::vector<int> coo_row_indices (control, coo->rowIndices, coo->num_nonzeros);
    clsparse::vector<int> coo_col_indices (control, coo->colIndices, coo->num_nonzeros);
    clsparse::vector<cl_double> coo_values (control, coo->values, coo->num_nonzeros);

    coo_col_indices = csr_col_indices;
    coo_values = csr_values;


    clsparseStatus status = offsets_to_indices(coo_row_indices, csr_row_offsets, csr->num_rows, control);

    CLSPARSE_V(status, "Error: offsets_to_indices");

    return status;
}
