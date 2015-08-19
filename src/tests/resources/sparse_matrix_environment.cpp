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