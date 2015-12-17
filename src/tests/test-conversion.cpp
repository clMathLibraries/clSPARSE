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

#if defined ( _WIN32 )
#define NOMINMAX
#endif

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

//boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/traits.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation_sparse.hpp>


clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

static cl_bool explicit_zeroes = true;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;

template <typename T>
class MatrixConversion : public ::testing::Test
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

public:
    void SetUp()
    {
#if 0
        // by default it is row_major;
        cldenseInitMatrix(&A);

        cl_int status;

        A.values = ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                    CSRE::n_cols * CSRE::n_rows * sizeof(T),
                                    nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        A.num_cols = CSRE::n_cols;
        A.num_rows = CSRE::n_rows;
        A.lead_dim = std::min(A.num_cols, A.num_rows);
#endif

    }

    void TearDown()
    {
#if 0
        ::clReleaseMemObject(A.values);

        cldenseInitMatrix(&A);
#endif
    }

    // uBLAS dense matrix format type
    typedef typename uBLAS::matrix<T, uBLAS::row_major, uBLAS::unbounded_array<T> > uBLASDenseM;

    cldenseMatrix A;



    void test_csr_to_dense()
    {
        
#if 1
        // by default it is row_major;
        cldenseInitMatrix(&A);

        cl_int status;

        A.values = ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
            CSRE::n_cols * CSRE::n_rows * sizeof(T),
            nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        A.num_cols = CSRE::n_cols;
        A.num_rows = CSRE::n_rows;
        A.lead_dim = std::min(A.num_cols, A.num_rows);
#endif
        
        if (typeid(T) == typeid(cl_float))
        {
            uBLASDenseM ublas_dense(CSRE::ublasSCsr);

            clsparseStatus status =
                    clsparseScsr2dense(&CSRE::csrSMatrix, &A, CLSE::control);

            EXPECT_EQ(clsparseSuccess, status);

            std::vector<T> result(ublas_dense.data().size());

            cl_int cl_status = clEnqueueReadBuffer(ClSparseEnvironment::queue,
                                                   this->A.values, 1, 0,
                                                   result.size()*sizeof(T),
                                                   result.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < ublas_dense.data().size(); i++)
            {
                // there should be exactly the same data
                ASSERT_NEAR(ublas_dense.data()[i], result[i], 1e-7);
            }

        }

        if (typeid(T) == typeid(cl_double))
        {
            uBLASDenseM ublas_dense(CSRE::ublasDCsr);

            clsparseStatus status =
                    clsparseDcsr2dense(&CSRE::csrDMatrix, &A, CLSE::control);

            EXPECT_EQ(clsparseSuccess, status);

            std::vector<T> result(ublas_dense.data().size());

            cl_int cl_status = clEnqueueReadBuffer(ClSparseEnvironment::queue,
                                                   this->A.values, 1, 0,
                                                   result.size()*sizeof(T),
                                                   result.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < ublas_dense.data().size(); i++)
            {
                // there should be exactly the same data
                ASSERT_NEAR(ublas_dense.data()[i], result[i], 1e-14);
            }
        }

#if 1
        ::clReleaseMemObject(A.values);

        cldenseInitMatrix(&A);
#endif
    }


    void test_dense_to_csr()
    {
#if 1
        // by default it is row_major;
        cldenseInitMatrix(&A);

        cl_int status;

        A.values = ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
            CSRE::n_cols * CSRE::n_rows * sizeof(T),
            nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        A.num_cols = CSRE::n_cols;
        A.num_rows = CSRE::n_rows;
        A.lead_dim = std::min(A.num_cols, A.num_rows);
#endif

        if (typeid(T) == typeid(cl_float))
        {
            //Create dense matrix;
            uBLASDenseM ublas_dense(CSRE::ublasSCsr);

            cl_int cl_status;

            //Create dense matrix on gpu; Shape is assigned in SetUp function;
            cl_status = ::clEnqueueWriteBuffer(CLSE::queue, A.values, CL_TRUE, 0,
                                               ublas_dense.data().size() * sizeof(T),
                                               ublas_dense.data().begin(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            //Create csr matrix container
            clsparseCsrMatrix csrMatx;
            clsparseInitCsrMatrix( &csrMatx );

            csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                               CSRE::ublasSCsr.value_data().size() * sizeof( T ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::ublasSCsr.index1_data().size() * sizeof( clsparseIdx_t ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::ublasSCsr.index2_data().size() * sizeof( clsparseIdx_t ), NULL, &cl_status );

            ASSERT_EQ(CL_SUCCESS, cl_status);

            // call conversion routine
            clsparseStatus status = clsparseSdense2csr(&A, &csrMatx, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            //Compare

            // Download GPU data
            std::vector<clsparseIdx_t> row_offsets(CSRE::ublasSCsr.index1_data().size());
            std::vector<clsparseIdx_t> col_indices(CSRE::ublasSCsr.index2_data().size());
            std::vector<T> values(CSRE::ublasSCsr.value_data().size());


            // Compare values
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.values, CL_TRUE, 0,
                                              values.size() * sizeof(T), values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < values.size(); i++)
                ASSERT_NEAR(CSRE::ublasSCsr.value_data()[i], values[i], 1e-7);


            // Compare row_offsets
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.rowOffsets, CL_TRUE, 0,
                                              row_offsets.size() * sizeof(clsparseIdx_t), row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (clsparseIdx_t i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(CSRE::ublasSCsr.index1_data()[i], row_offsets[i]);


            // Compare col indices
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.colIndices, CL_TRUE, 0,
                                              col_indices.size() * sizeof(clsparseIdx_t), col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(CSRE::ublasSCsr.index2_data()[i], col_indices[i]);

            // Release csrMatrix data
            cl_status = ::clReleaseMemObject(csrMatx.values);
            cl_status = ::clReleaseMemObject(csrMatx.colIndices);
            cl_status = ::clReleaseMemObject(csrMatx.rowOffsets);

        }

        /* There is no dense to csr for double precision */
        if (typeid(T) == typeid(cl_double))
        {
            //Create dense matrix;
            uBLASDenseM ublas_dense(CSRE::ublasDCsr);

            cl_int cl_status;

            //Create dense matrix on gpu; Shape is assigned in SetUp function;
            cl_status = ::clEnqueueWriteBuffer(CLSE::queue, A.values, CL_TRUE, 0,
                                               ublas_dense.data().size() * sizeof(T),
                                               ublas_dense.data().begin(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            //Create csr matrix container
            clsparseCsrMatrix csrMatx;
            clsparseInitCsrMatrix( &csrMatx );

            csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                               CSRE::csrDMatrix.num_nonzeros * sizeof( T ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   ( CSRE::csrDMatrix.num_rows + 1 ) * sizeof( clsparseIdx_t ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::csrDMatrix.num_nonzeros * sizeof( clsparseIdx_t ), NULL, &cl_status );

            ASSERT_EQ(CL_SUCCESS, cl_status);

            // call conversion routine
            clsparseStatus status = clsparseDdense2csr(&A, &csrMatx, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            //Compare

            // Download GPU data
            std::vector<clsparseIdx_t> row_offsets(CSRE::csrDMatrix.num_rows + 1);
            std::vector<clsparseIdx_t> col_indices(CSRE::csrDMatrix.num_nonzeros);
            std::vector<T> values (CSRE::csrDMatrix.num_nonzeros);

            // Compare values
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.values, CL_TRUE, 0,
                                              values.size() * sizeof(T), values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < values.size(); i++)
            {
                EXPECT_DOUBLE_EQ(CSRE::ublasDCsr.value_data()[i], values[i]);
            }


            // Compare row_offsets
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.rowOffsets, CL_TRUE, 0,
                                              row_offsets.size() * sizeof(clsparseIdx_t), row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (clsparseIdx_t i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(CSRE::ublasDCsr.index1_data()[i], row_offsets[i]);


            // Compare col indices
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.colIndices, CL_TRUE, 0,
                                              col_indices.size() * sizeof(clsparseIdx_t), col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(CSRE::ublasDCsr.index2_data()[i], col_indices[i]);

            // Release csrMatrix data
            cl_status = ::clReleaseMemObject(csrMatx.values);
            cl_status = ::clReleaseMemObject(csrMatx.colIndices);
            cl_status = ::clReleaseMemObject(csrMatx.rowOffsets);

        }

#if 0
        ::clReleaseMemObject(A.values);

        cldenseInitMatrix(&A);
#endif

    }

    void test_coo_to_csr()
    {

        clsparseIdx_t nnz, num_rows, num_cols;
        clsparseStatus status;
        cl_int cl_status;

        // Prepare COO matrix from input file
        status = clsparseHeaderfromFile(&nnz, &num_rows, &num_cols,
                                        CSRE::file_name.c_str());
        ASSERT_EQ(clsparseSuccess, status);

        clsparseCooMatrix cooMatrix;
        status = clsparseInitCooMatrix (&cooMatrix);
        ASSERT_EQ(clsparseSuccess, status);

        cooMatrix.num_nonzeros = nnz;
        cooMatrix.num_cols = num_cols;
        cooMatrix.num_rows = num_rows;

        cooMatrix.colIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_ONLY,
                                 cooMatrix.num_nonzeros * sizeof(clsparseIdx_t),
                                 NULL, &cl_status);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        cooMatrix.rowIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_ONLY,
                                 cooMatrix.num_nonzeros * sizeof(clsparseIdx_t),
                                 NULL, &cl_status);
        ASSERT_EQ(CL_SUCCESS, cl_status);


        // Now we have to distinct types
        if(typeid(T) == typeid(cl_float))
        {

            cooMatrix.values =
                    ::clCreateBuffer(CLSE::context, CL_MEM_READ_ONLY,
                                     cooMatrix.num_nonzeros * sizeof(T),
                                     NULL, &cl_status);
            ASSERT_EQ (CL_SUCCESS, cl_status);

            status = clsparseSCooMatrixfromFile(&cooMatrix,
                                                CSRE::file_name.c_str(),
                                                CLSE::control,
                                                explicit_zeroes);
            ASSERT_EQ(clsparseSuccess, status);

            // To save memory let us use the CSRE gpu matrix which have the
            // proper shape already.
            status = clsparseScoo2csr(&cooMatrix, &CSRE::csrSMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Compare newly generated results with the uBLAS matrix from CSRE
            std::vector<clsparseIdx_t> row_offsets(CSRE::csrSMatrix.num_rows + 1);
            std::vector<clsparseIdx_t> col_indices(CSRE::csrSMatrix.num_nonzeros);
            std::vector<T> values (CSRE::csrSMatrix.num_nonzeros);

            //Download GPU data to vectors;
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.values,
                                              CL_TRUE, 0, values.size() * sizeof(T),
                                              values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare values;
            for (clsparseIdx_t i = 0; i < values.size(); i++)
                EXPECT_FLOAT_EQ(values[i], CSRE::ublasSCsr.value_data()[i]);

            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.colIndices,
                                              CL_TRUE, 0, col_indices.size() * sizeof(clsparseIdx_t),
                                              col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare column indices
            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(col_indices[i], CSRE::ublasSCsr.index2_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.rowOffsets,
                                              CL_TRUE, 0, row_offsets.size() * sizeof(clsparseIdx_t),
                                              row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare row offsets
            for (clsparseIdx_t i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(row_offsets[i], CSRE::ublasSCsr.index1_data()[i]);
        }

        if (typeid(T) == typeid(cl_double))
        {
            cooMatrix.values =
                    ::clCreateBuffer(CLSE::context, CL_MEM_READ_ONLY,
                                     cooMatrix.num_nonzeros * sizeof(T),
                                     NULL, &cl_status);
            ASSERT_EQ (CL_SUCCESS, cl_status);

            status = clsparseDCooMatrixfromFile(&cooMatrix,
                                                CSRE::file_name.c_str(),
                                                CLSE::control,
                                                explicit_zeroes);
            ASSERT_EQ(clsparseSuccess, status);

            // To save memory let us use the CSRE gpu matrix which have the
            // proper shape already.
            status = clsparseDcoo2csr(&cooMatrix, &CSRE::csrDMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Compare newly generated results with the uBLAS matrix from CSRE
            std::vector<clsparseIdx_t> row_offsets(CSRE::csrDMatrix.num_rows + 1);
            std::vector<clsparseIdx_t> col_indices(CSRE::csrDMatrix.num_nonzeros);
            std::vector<T> values (CSRE::csrDMatrix.num_nonzeros);

            //Download GPU data to vectors;
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.values,
                                              CL_TRUE, 0, values.size() * sizeof(T),
                                              values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare values;
            for (clsparseIdx_t i = 0; i < values.size(); i++)
                EXPECT_DOUBLE_EQ(values[i], CSRE::ublasDCsr.value_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.colIndices,
                                              CL_TRUE, 0, col_indices.size() * sizeof(clsparseIdx_t),
                                              col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare column indices
            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(col_indices[i], CSRE::ublasDCsr.index2_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.rowOffsets,
                                              CL_TRUE, 0, row_offsets.size() * sizeof(clsparseIdx_t),
                                              row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare row offsets
            for (clsparseIdx_t i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(row_offsets[i], CSRE::ublasDCsr.index1_data()[i]);
        }


        cl_status = ::clReleaseMemObject(cooMatrix.colIndices);
        ASSERT_EQ(CL_SUCCESS, cl_status);
        cl_status = ::clReleaseMemObject(cooMatrix.rowIndices);
        ASSERT_EQ(CL_SUCCESS, cl_status);
        cl_status = ::clReleaseMemObject(cooMatrix.values);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        clsparseInitCooMatrix(&cooMatrix);
    }

    void test_csr_to_coo()
    {
        clsparseStatus status;
        cl_int cl_status;
        // Create coo matrix on GPU;
        clsparseCooMatrix cooMatrix;

        status = clsparseInitCooMatrix (&cooMatrix);

        ASSERT_EQ(clsparseSuccess, status);

        cooMatrix.colIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                 CSRE::csrSMatrix.num_nonzeros * sizeof(clsparseIdx_t),
                                 nullptr, &cl_status);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        cooMatrix.rowIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                 CSRE::csrSMatrix.num_nonzeros * sizeof(clsparseIdx_t),
                                 nullptr, &cl_status);

        ASSERT_EQ(CL_SUCCESS, cl_status);

        // Distinct types
        if (typeid(T) == typeid(cl_float))
        {
            cooMatrix.values =
                    ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                     CSRE::csrSMatrix.num_nonzeros * sizeof(T),
                                     nullptr, &cl_status);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            status = clsparseScsr2coo(&CSRE::csrSMatrix, &cooMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Generate reference.
            float* vals = (float*)&CSRE::ublasSCsr.value_data()[0];
            clsparseIdx_t* rows = &CSRE::ublasSCsr.index1_data()[0];
            clsparseIdx_t* cols = &CSRE::ublasSCsr.index2_data()[0];

            clsparseIdx_t* coo_rows = new clsparseIdx_t[CSRE::n_vals];
            clsparseIdx_t* coo_cols = new clsparseIdx_t[CSRE::n_vals];
            float* coo_vals = new float[CSRE::n_vals];
            clsparseIdx_t total_vals = 0;
            for ( clsparseIdx_t row = 0; row < CSRE::n_rows; row++)
            {
                for ( clsparseIdx_t i = rows[row]; i < rows[row + 1]; i++)
                {
                    coo_rows[total_vals] = row;
                    coo_cols[total_vals] = cols[i];
                    coo_vals[total_vals] = vals[i];
                    total_vals++;
                }
            }

            // Compare result

            // Download results from GPU
            std::vector<clsparseIdx_t> row_indices(cooMatrix.num_nonzeros);
            std::vector<clsparseIdx_t> col_indices(cooMatrix.num_nonzeros);
            std::vector<T> values(cooMatrix.num_nonzeros);

            // row indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.rowIndices,
                                            CL_TRUE, 0, row_indices.size() * sizeof( clsparseIdx_t ),
                                            row_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < row_indices.size(); i++)
                ASSERT_EQ(coo_rows[i], row_indices[i]);

            // col indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.colIndices,
                                            CL_TRUE, 0, col_indices.size() * sizeof( clsparseIdx_t ),
                                            col_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(coo_cols[i], col_indices[i]);


            // values
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.values,
                                            CL_TRUE, 0, values.size() * sizeof(T),
                                            values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < values.size(); i++)
                EXPECT_FLOAT_EQ(coo_vals[i], values[i]);

            delete[] coo_rows;
            delete[] coo_cols;
            delete[] coo_vals;
        }

        if (typeid(T) == typeid(cl_double))
        {
            cooMatrix.values =
                    ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                     CSRE::csrDMatrix.num_nonzeros * sizeof(T),
                                     nullptr, &cl_status);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            status = clsparseDcsr2coo(&CSRE::csrDMatrix, &cooMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Generate reference;
            double* vals = (double*)&CSRE::ublasDCsr.value_data()[0];
            clsparseIdx_t* rows = &CSRE::ublasDCsr.index1_data()[0];
            clsparseIdx_t* cols = &CSRE::ublasDCsr.index2_data()[0];

            clsparseIdx_t* coo_rows = new clsparseIdx_t[CSRE::n_vals];
            clsparseIdx_t* coo_cols = new clsparseIdx_t[CSRE::n_vals];
            double* coo_vals = new double[CSRE::n_vals];
            clsparseIdx_t total_vals = 0;
            for ( clsparseIdx_t row = 0; row < CSRE::n_rows; row++)
            {
                for ( clsparseIdx_t i = rows[row]; i < rows[row + 1]; i++)
                {
                    coo_rows[total_vals] = row;
                    coo_cols[total_vals] = cols[i];
                    coo_vals[total_vals] = vals[i];
                    total_vals++;
                }
            }

            // Compare result

            // Download results from GPU
            std::vector<clsparseIdx_t> row_indices(cooMatrix.num_nonzeros);
            std::vector<clsparseIdx_t> col_indices(cooMatrix.num_nonzeros);
            std::vector<T> values(cooMatrix.num_nonzeros);


            // row indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.rowIndices,
                                            CL_TRUE, 0, row_indices.size() * sizeof( clsparseIdx_t ),
                                            row_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < row_indices.size(); i++)
                ASSERT_EQ(coo_rows[i], row_indices[i]);

            // col indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.colIndices,
                                            CL_TRUE, 0, col_indices.size() * sizeof( clsparseIdx_t ),
                                            col_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(coo_cols[i], col_indices[i]);


            // values
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.values,
                                            CL_TRUE, 0, values.size() * sizeof(T),
                                            values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (clsparseIdx_t i = 0; i < values.size(); i++)
                EXPECT_DOUBLE_EQ(coo_vals[i], values[i]);

            delete[] coo_rows;
            delete[] coo_cols;
            delete[] coo_vals;
        }

        cl_status = ::clReleaseMemObject(cooMatrix.colIndices);
        ASSERT_EQ(CL_SUCCESS, cl_status);
        cl_status = ::clReleaseMemObject(cooMatrix.rowIndices);
        ASSERT_EQ(CL_SUCCESS, cl_status);
        cl_status = ::clReleaseMemObject(cooMatrix.values);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        clsparseInitCooMatrix(&cooMatrix);
    }

};

//typedef ::testing::Types<cl_float, cl_double> TYPES;
typedef ::testing::Types<cl_float, cl_double> TYPES;
TYPED_TEST_CASE(MatrixConversion, TYPES);

TYPED_TEST(MatrixConversion, csr_to_dense)
{
    this->test_csr_to_dense();
}

TYPED_TEST(MatrixConversion, dense_to_csr)
{
    this->test_dense_to_csr();
}

TYPED_TEST(MatrixConversion, csr_to_coo)
{
    this->test_csr_to_coo();
}

TYPED_TEST(MatrixConversion, coo_to_csr)
{
    this->test_coo_to_csr();
}



int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    std::string path;
//    double alpha;
//    double beta;
    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.")
            ("no_zeroes,z", po::bool_switch()->default_value(false),
             "Disable reading explicit zeroes from the input matrix market file.");

    //	Parse the command line options, ignore unrecognized options and collect them into a vector of strings
    //  Googletest itself accepts command line flags that we wish to pass further on
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store( parsed, vm );
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
        po::notify( vm );
    }
    catch( po::error& error )
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what( ) << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    std::vector< std::string > to_pass_further = po::collect_unrecognized( parsed.options, po::include_positional );

    //check platform
    if(vm.count("platform"))
    {
        if ("AMD" == platform)
        {
            pID = AMD;
        }
        else if ("NVIDIA" == platform)
        {
            pID = NVIDIA;
        }
        else
        {

            std::cout << "The platform option is missing or is ill defined!\n";
            std::cout << "Given [" << platform << "]" << std::endl;
            platform = "AMD";
            pID = AMD;
            std::cout << "Setting [" << platform << "] as default" << std::endl;
        }

    }

    if (vm["no_zeroes"].as<bool>())
        explicit_zeroes = false;

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, 1, 0,
                                                  CLSE::queue, CLSE::context, explicit_zeroes ));
    return RUN_ALL_TESTS();
}
