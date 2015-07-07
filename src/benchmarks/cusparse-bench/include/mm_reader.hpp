#pragma once
#ifndef CUBLAS_BENCHMARK_MM_READER_HXX__
#define CUBLAS_BENCHMARK_MM_READER_HXX__

#include <vector>

int
cooMatrixfromFile( std::vector< int >& row_indices, std::vector< int >& col_indices,
std::vector< float >& values, const char* filePath );

int
csrMatrixfromFile( std::vector< int >& row_offsets, std::vector< int >& col_indices,
std::vector< float >& values, const char* filePath );

int
csrMatrixfromFile(std::vector< int >& row_offsets, std::vector< int >& col_indices,
std::vector< double >& values, const char* filePath);

int sparseHeaderfromFile(int* nnz, int* rows, int* cols, const char* filePath);

#endif