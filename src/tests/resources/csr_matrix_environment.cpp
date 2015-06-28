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
