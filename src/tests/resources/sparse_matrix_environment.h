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
#ifndef _SPARSE_MATRIX_ENVIRONMENT_H_
#define _SPARSE_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "clsparse_environment.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <algorithm>
#include <cfloat>

//#define TEST_LONG
//#define NORMALIZE

using CLSE = ClSparseEnvironment;

namespace uBLAS = boost::numeric::ublas;

/**
* @brief The CSRSparseEnvironment class will have the input parameters for SpMSpM tests
* They are list of csr matrices in csr format in mtx files.
*/
// Currently only single precision is considered
class CSRSparseEnvironment : public ::testing::Environment {
public:
    using sMatrixType = uBLAS::compressed_matrix<float, uBLAS::row_major, 0, uBLAS::unbounded_array<clsparseIdx_t> >;
    //using dMatrixType = uBLAS::compressed_matrix<double, uBLAS::row_major, 0, uBLAS::unbounded_array<size_t> >;

    explicit CSRSparseEnvironment(const std::string& path, cl_command_queue queue, cl_context context, cl_bool explicit_zeroes = true)
        : queue(queue), context(context)
    {
        file_name = path;
        clsparseStatus read_status = clsparseHeaderfromFile(&n_vals, &n_rows, &n_cols, file_name.c_str());
        if (read_status)
        {
            exit(-3);
        }

        clsparseInitCsrMatrix(&csrSMatrix);
        csrSMatrix.num_nonzeros = n_vals;
        csrSMatrix.num_rows = n_rows;
        csrSMatrix.num_cols = n_cols;

        //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;
        csrSMatrix.values = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.num_nonzeros * sizeof(cl_float), NULL, &status);

        csrSMatrix.colIndices = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrix.num_nonzeros * sizeof(cl_int), NULL, &status);

        csrSMatrix.rowOffsets = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int), NULL, &status);

        clsparseStatus fileError = clsparseSCsrMatrixfromFile(&csrSMatrix, file_name.c_str(), CLSE::control, explicit_zeroes);
        if (fileError != clsparseSuccess)
            throw std::runtime_error("Could not read matrix market data from disk");

        clsparseCsrMetaCompute( &csrSMatrix, CLSE::control );

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        ublasSCsr = sMatrixType(n_rows, n_cols, n_vals);

        // This is nasty. Without that call ublasSCsr is not working correctly.
        ublasSCsr.complete_index1_data();

        // copy host matrix arrays to device;
        cl_int copy_status;

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.values, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_float),
            ublasSCsr.value_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.rowOffsets, CL_TRUE, 0,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int),
            ublasSCsr.index1_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueReadBuffer(queue, csrSMatrix.colIndices, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_int),
            ublasSCsr.index2_data().begin(),
            0, NULL, NULL);

#ifdef NORMALIZE

         int size = csrSMatrix.num_nonzeros;

         float min = *std::min_element(ublasSCsr.value_data().begin(), ublasSCsr.value_data().begin()+size);
         float max = *std::max_element(ublasSCsr.value_data().begin(), ublasSCsr.value_data().begin()+size);
         std::cout << "min ="  << min << std::endl;
         std::cout << "max ="  << max << std::endl;

         for(int i=0; i<size; i++)
         {
          ublasSCsr.value_data()[i] = (ublasSCsr.value_data()[i]-min)/(max-min);
          //std::cout <<  ublasSCsr.value_data()[i] << " ";
         }
         std::cout << std::endl;
  
         copy_status = clEnqueueWriteBuffer(queue, csrSMatrix.values, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_float),
            ublasSCsr.value_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueWriteBuffer(queue, csrSMatrix.rowOffsets, CL_TRUE, 0,
            (csrSMatrix.num_rows + 1) * sizeof(cl_int),
            ublasSCsr.index1_data().begin(),
            0, NULL, NULL);

        copy_status = clEnqueueWriteBuffer(queue, csrSMatrix.colIndices, CL_TRUE, 0,
            csrSMatrix.num_nonzeros * sizeof(cl_int),
            ublasSCsr.index2_data().begin(),
            0, NULL, NULL);

#endif

#ifdef TEST_LONG

       int m = 1;
       int k = 2;
       int n = 30720;
       int nnzA = 2;
       int nnzB = 30720 ;

       ublasSCsrA = sMatrixType(m, k, nnzA);
       ublasSCsrB = sMatrixType(k, n, nnzB);

       // This is nasty. Without that call ublasSCsr is not working correctly.
       ublasSCsrA.complete_index1_data();
       ublasSCsrB.complete_index1_data();
  
       ublasSCsrA.index1_data()[0] = 0;
       ublasSCsrA.index1_data()[1] = nnzA;

       for (int i = 0; i < nnzA; i++)
       {
           ublasSCsrA.index2_data()[i] = i;
           ublasSCsrA.value_data()[i] = 1;
       }

       // B

       ublasSCsrB.index1_data()[0] = 0;
       ublasSCsrB.index1_data()[1] = nnzB/2;
       ublasSCsrB.index1_data()[2] = nnzB;

       for (int i = 0; i < nnzB/2; i++)
       {
          ublasSCsrB.index2_data()[i] = 2 * i;
          //ublasSCsrB.value_data()[i] = 2 * i ;
          ublasSCsrB.value_data()[i] = 1 ;
       }

       for (int i = 0; i < nnzB/2; i++)
       {
          ublasSCsrB.index2_data()[nnzB/2 + i] = 2 * i + 1;
          ublasSCsrB.value_data()[nnzB/2 + i] = 2 * i + 1;
       }
 

       //device A
       clsparseInitCsrMatrix(&csrSMatrixA);
       csrSMatrixA.num_nonzeros = nnzA;
       csrSMatrixA.num_rows = m;
       csrSMatrixA.num_cols = k;
       size_t metaSize;
       clsparseCsrMetaSize(&csrSMatrixA, CLSE::control, &metaSize );

       //  Load single precision data from file; this API loads straight into GPU memory
       csrSMatrixA.values = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrixA.num_nonzeros * sizeof(cl_float), NULL, &status);

       csrSMatrixA.colIndices = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrixA.num_nonzeros * sizeof(cl_int), NULL, &status);

       csrSMatrixA.rowOffsets = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            (csrSMatrixA.num_rows + 1) * sizeof(cl_int), NULL, &status);

       //load data to device
       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixA.values, CL_TRUE, 0,
            csrSMatrixA.num_nonzeros * sizeof(cl_float),
            ublasSCsrA.value_data().begin(),
            0, NULL, NULL);

       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixA.rowOffsets, CL_TRUE, 0,
            (csrSMatrixA.num_rows + 1) * sizeof(cl_int),
            ublasSCsrA.index1_data().begin(),
            0, NULL, NULL);

       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixA.colIndices, CL_TRUE, 0,
            csrSMatrixA.num_nonzeros * sizeof(cl_int),
            ublasSCsrA.index2_data().begin(),
            0, NULL, NULL);

        //device B
        clsparseInitCsrMatrix(&csrSMatrixB);
        csrSMatrixB.num_nonzeros = nnzB;
        csrSMatrixB.num_rows = k;
        csrSMatrixB.num_cols = n;
        clsparseCsrMetaSize(&csrSMatrixB, CLSE::control, &metaSize );

        //  Load single precision data from file; this API loads straight into GPU memory
        csrSMatrixB.values = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrixB.num_nonzeros * sizeof(cl_float), NULL, &status);

        csrSMatrixB.colIndices = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            csrSMatrixB.num_nonzeros * sizeof(cl_int), NULL, &status);

        csrSMatrixB.rowOffsets = ::clCreateBuffer(context, CL_MEM_READ_ONLY,
            (csrSMatrixB.num_rows + 1) * sizeof(cl_int), NULL, &status);

       //load data to device
       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixB.values, CL_TRUE, 0,
            csrSMatrixB.num_nonzeros * sizeof(cl_float),
            ublasSCsrB.value_data().begin(),
            0, NULL, NULL);

       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixB.rowOffsets, CL_TRUE, 0,
            (csrSMatrixB.num_rows + 1) * sizeof(cl_int),
            ublasSCsrB.index1_data().begin(),
            0, NULL, NULL);

       copy_status = clEnqueueWriteBuffer(queue, csrSMatrixB.colIndices, CL_TRUE, 0,
            csrSMatrixB.num_nonzeros * sizeof(cl_int),
            ublasSCsrB.index2_data().begin(),
            0, NULL, NULL);


#endif

        if (copy_status)
        {
            TearDown();
            exit(-5);
        }
    }// end C'tor

    void SetUp()
    {
        // Prepare data to it's default state
    }

    //cleanup
    void TearDown()
    {
    }

    std::string getFileName()
    {
        return file_name;
    }

    ~CSRSparseEnvironment()
    {
        //release buffers;
        ::clReleaseMemObject(csrSMatrix.values);
        ::clReleaseMemObject(csrSMatrix.colIndices);
        ::clReleaseMemObject(csrSMatrix.rowOffsets);

        //bring csrSMatrix  to its initial state
        clsparseInitCsrMatrix(&csrSMatrix);
    }
        

    static sMatrixType ublasSCsr;
    //static sMatrixType ublasCsrB;
    //static sMatrixType ublasCsrC;    

    static clsparseIdx_t n_rows;
    static clsparseIdx_t n_cols;
    static clsparseIdx_t n_vals;

    //cl buffers ;
    static clsparseCsrMatrix csrSMatrix; // input 1

#ifdef TEST_LONG
    static clsparseCsrMatrix csrSMatrixA;
    static clsparseCsrMatrix csrSMatrixB;
    static sMatrixType ublasSCsrA;
    static sMatrixType ublasSCsrB;
#endif
    //static clsparseCsrMatrix csrMatrixC; // output

    static std::string file_name;

private:
    cl_command_queue queue;
    cl_context context;
};


#endif // _SPARSE_MATRIX_ENVIRONMENT_H_
