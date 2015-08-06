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