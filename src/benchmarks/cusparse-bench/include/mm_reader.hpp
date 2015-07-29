#pragma once
#ifndef CUBLAS_BENCHMARK_MM_READER_HXX__
#define CUBLAS_BENCHMARK_MM_READER_HXX__

#include <vector>

int sparseHeaderfromFile( int* nnz, int* rows, int* cols, const char* filePath );

template< class T > int
cooMatrixfromFile( std::vector< int >& row_indices, std::vector< int >& col_indices,
std::vector< T >& values, const char* filePath );

template< class T > int
csrMatrixfromFile( std::vector< int >& row_offsets, std::vector< int >& col_indices,
std::vector< T >& values, const char* filePath );

#endif