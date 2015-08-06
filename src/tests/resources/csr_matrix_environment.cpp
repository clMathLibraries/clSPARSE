/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
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

#include "csr_matrix_environment.h"

CSREnvironment::sMatrixType CSREnvironment::ublasSCsr = CSREnvironment::sMatrixType();

CSREnvironment::dMatrixType CSREnvironment::ublasDCsr = CSREnvironment::dMatrixType();


cl_int CSREnvironment::n_rows = 0;
cl_int CSREnvironment::n_cols = 0;
cl_int CSREnvironment::n_vals = 0;

clsparseCsrMatrix CSREnvironment::csrSMatrix = clsparseCsrMatrix();
clsparseCsrMatrix CSREnvironment::csrDMatrix = clsparseCsrMatrix();


cl_double CSREnvironment::alpha = 1.0;
cl_double CSREnvironment::beta = 0.0;

std::string CSREnvironment::file_name = std::string();
