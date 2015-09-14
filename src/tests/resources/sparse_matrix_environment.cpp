<<<<<<< HEAD
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
=======
>>>>>>> merged SpMSpM code
#include "sparse_matrix_environment.h"

CSRSparseEnvironment::sMatrixType CSRSparseEnvironment::ublasSCsr = CSRSparseEnvironment::sMatrixType();

#ifdef TEST_LONG
CSRSparseEnvironment::sMatrixType CSRSparseEnvironment::ublasSCsrA = CSRSparseEnvironment::sMatrixType();
CSRSparseEnvironment::sMatrixType CSRSparseEnvironment::ublasSCsrB = CSRSparseEnvironment::sMatrixType();
#endif

cl_int CSRSparseEnvironment::n_rows = 0;
cl_int CSRSparseEnvironment::n_cols = 0;
cl_int CSRSparseEnvironment::n_vals = 0;

clsparseCsrMatrix CSRSparseEnvironment::csrSMatrix = clsparseCsrMatrix();

#ifdef TEST_LONG
clsparseCsrMatrix CSRSparseEnvironment::csrSMatrixA = clsparseCsrMatrix();
clsparseCsrMatrix CSRSparseEnvironment::csrSMatrixB = clsparseCsrMatrix();

#endif

std::string CSRSparseEnvironment::file_name = std::string();