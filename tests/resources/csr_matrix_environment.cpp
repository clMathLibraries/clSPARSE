#include "csr_matrix_environment.h"

std::vector<int> CSREnvironment::row_offsets = std::vector<int>();
std::vector<int> CSREnvironment::col_indices = std::vector<int>();
std::vector<float> CSREnvironment::f_values = std::vector<float>();
std::vector<double> CSREnvironment::d_values = std::vector<double>();
int CSREnvironment::n_rows = 0;
int CSREnvironment::n_cols = 0;
int CSREnvironment::n_vals = 0;

cl_mem CSREnvironment::cl_row_offsets = NULL;
cl_mem CSREnvironment::cl_col_indices = NULL;
cl_mem CSREnvironment::cl_f_values = NULL;
cl_mem CSREnvironment::cl_d_values = NULL;

