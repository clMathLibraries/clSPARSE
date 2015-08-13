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

    }

    void TearDown()
    {

        ::clReleaseMemObject(A.values);

        cldenseInitMatrix(&A);
    }

    // uBLAS dense matrix format type
    typedef typename uBLAS::matrix<T, uBLAS::row_major, uBLAS::unbounded_array<T> > uBLASDenseM;

    // uBLAS coo matrix format type;
    typedef typename uBLAS::coordinate_matrix<T,  uBLAS::row_major, 0, uBLAS::unbounded_array<cl_ulong> > uBLASCooM;


    cldenseMatrix A;



    void test_csr_to_dense()
    {
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

            for(int i = 0; i < ublas_dense.data().size(); i++)
            {
                // there should be exactly the same data
                ASSERT_EQ(ublas_dense.data()[i], result[i]);
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

            for(int i = 0; i < ublas_dense.data().size(); i++)
            {
                // there should be exactly the same data
                ASSERT_EQ(ublas_dense.data()[i], result[i]);
            }
        }
    }


    void test_dense_to_csr()
    {
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
                                                   CSRE::ublasSCsr.index1_data().size() * sizeof( cl_int ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::ublasSCsr.index2_data().size() * sizeof( cl_int ), NULL, &cl_status );

            ASSERT_EQ(CL_SUCCESS, cl_status);

            // call conversion routine
            clsparseStatus status = clsparseSdense2csr(&csrMatx, &A, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            //Compare

            // Download GPU data
            std::vector<cl_int> row_offsets(CSRE::ublasSCsr.index1_data().size());
            std::vector<cl_int> col_indices(CSRE::ublasSCsr.index2_data().size());
            std::vector<T> values(CSRE::ublasSCsr.value_data().size());


            // Compare values
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.values, CL_TRUE, 0,
                                              values.size() * sizeof(T), values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(CSRE::ublasSCsr.value_data()[i], values[i]);


            // Compare row_offsets
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.rowOffsets, CL_TRUE, 0,
                                              row_offsets.size() * sizeof(cl_int), row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (int i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(CSRE::ublasSCsr.index1_data()[i], row_offsets[i]);


            // Compare col indices
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.colIndices, CL_TRUE, 0,
                                              col_indices.size() * sizeof(cl_int), col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (int i = 0; i < col_indices.size(); i++)
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
                                               CSRE::ublasDCsr.value_data().size() * sizeof( T ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::ublasDCsr.index1_data().size() * sizeof( cl_int ), NULL, &cl_status );
            ASSERT_EQ(CL_SUCCESS, cl_status);

            csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                                   CSRE::ublasDCsr.index2_data().size() * sizeof( cl_int ), NULL, &cl_status );

            ASSERT_EQ(CL_SUCCESS, cl_status);

            // call conversion routine
            clsparseStatus status = clsparseDdense2csr(&csrMatx, &A, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            //Compare

            // Download GPU data
            std::vector<cl_int> row_offsets(CSRE::ublasDCsr.index1_data().size());
            std::vector<cl_int> col_indices(CSRE::ublasDCsr.index2_data().size());
            std::vector<T> values(CSRE::ublasDCsr.value_data().size());


            // Compare values
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.values, CL_TRUE, 0,
                                              values.size() * sizeof(T), values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(CSRE::ublasDCsr.value_data()[i], values[i]);


            // Compare row_offsets
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.rowOffsets, CL_TRUE, 0,
                                              row_offsets.size() * sizeof(cl_int), row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (int i = 0; i < row_offsets.size(); i++)
                ASSERT_EQ(CSRE::ublasDCsr.index1_data()[i], row_offsets[i]);


            // Compare col indices
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, csrMatx.colIndices, CL_TRUE, 0,
                                              col_indices.size() * sizeof(cl_int), col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);


            for (int i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(CSRE::ublasDCsr.index2_data()[i], col_indices[i]);

            // Release csrMatrix data
            cl_status = ::clReleaseMemObject(csrMatx.values);
            cl_status = ::clReleaseMemObject(csrMatx.colIndices);
            cl_status = ::clReleaseMemObject(csrMatx.rowOffsets);

        }
    }

    void test_coo_to_csr()
    {

        cl_int nnz, num_rows, num_cols;
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
                                 cooMatrix.num_nonzeros * sizeof(cl_int),
                                 NULL, &cl_status);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        cooMatrix.rowIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_ONLY,
                                 cooMatrix.num_nonzeros * sizeof(cl_int),
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
                                                CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);

            // To save memory let us use the CSRE gpu matrix which have the
            // proper shape already.
            status = clsparseScoo2csr(&cooMatrix, &CSRE::csrSMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Compare newly generated results with the uBLAS matrix from CSRE
            std::vector<cl_int> row_offsets(CSRE::csrSMatrix.num_rows + 1);
            std::vector<cl_int> col_indices(CSRE::csrSMatrix.num_nonzeros);
            std::vector<T> values (CSRE::csrSMatrix.num_nonzeros);

            //Download GPU data to vectors;
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.values,
                                              CL_TRUE, 0, values.size() * sizeof(T),
                                              values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare values;
            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(values[i], CSRE::ublasSCsr.value_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.colIndices,
                                              CL_TRUE, 0, col_indices.size() * sizeof(cl_int),
                                              col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare column indices
            for (int i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(col_indices[i], CSRE::ublasSCsr.index2_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrSMatrix.rowOffsets,
                                              CL_TRUE, 0, row_offsets.size() * sizeof(cl_int),
                                              row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare row offsets
            for (int i = 0; i < row_offsets.size(); i++)
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
                                                CLSE::control);
            ASSERT_EQ(clsparseSuccess, status);

            // To save memory let us use the CSRE gpu matrix which have the
            // proper shape already.
            status = clsparseDcoo2csr(&cooMatrix, &CSRE::csrDMatrix, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            // Compare newly generated results with the uBLAS matrix from CSRE
            std::vector<cl_int> row_offsets(CSRE::csrDMatrix.num_rows + 1);
            std::vector<cl_int> col_indices(CSRE::csrDMatrix.num_nonzeros);
            std::vector<T> values (CSRE::csrDMatrix.num_nonzeros);

            //Download GPU data to vectors;
            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.values,
                                              CL_TRUE, 0, values.size() * sizeof(T),
                                              values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare values;
            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(values[i], CSRE::ublasDCsr.value_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.colIndices,
                                              CL_TRUE, 0, col_indices.size() * sizeof(cl_int),
                                              col_indices.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare column indices
            for (int i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(col_indices[i], CSRE::ublasDCsr.index2_data()[i]);


            cl_status = ::clEnqueueReadBuffer(CLSE::queue, CSRE::csrDMatrix.rowOffsets,
                                              CL_TRUE, 0, row_offsets.size() * sizeof(cl_int),
                                              row_offsets.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            // Compare row offsets
            for (int i = 0; i < row_offsets.size(); i++)
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
                                 CSRE::csrSMatrix.num_nonzeros * sizeof(cl_int),
                                 nullptr, &cl_status);
        ASSERT_EQ(CL_SUCCESS, cl_status);

        cooMatrix.rowIndices =
                ::clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                 CSRE::csrSMatrix.num_nonzeros * sizeof(cl_int),
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

            // Generate reference;
            uBLASCooM ublas_coo(CSRE::ublasSCsr);

            // Compare result

            // Download results from GPU
            std::vector<cl_int> row_indices(cooMatrix.num_nonzeros);
            std::vector<cl_int> col_indices(cooMatrix.num_nonzeros);
            std::vector<T> values(cooMatrix.num_nonzeros);

            // row indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.rowIndices,
                                            CL_TRUE, 0, row_indices.size() * sizeof(cl_int),
                                            row_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < row_indices.size(); i++)
                ASSERT_EQ(ublas_coo.index1_data()[i], row_indices[i]);

            // col indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.colIndices,
                                            CL_TRUE, 0, col_indices.size() * sizeof(cl_int),
                                            col_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(ublas_coo.index2_data()[i], col_indices[i]);


            // values
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.values,
                                            CL_TRUE, 0, values.size() * sizeof(T),
                                            values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(ublas_coo.value_data()[i], values[i]);

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
            uBLASCooM ublas_coo(CSRE::ublasDCsr);

            // Compare result

            // Download results from GPU
            std::vector<cl_int> row_indices(cooMatrix.num_nonzeros);
            std::vector<cl_int> col_indices(cooMatrix.num_nonzeros);
            std::vector<T> values(cooMatrix.num_nonzeros);


            // row indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.rowIndices,
                                            CL_TRUE, 0, row_indices.size() * sizeof(cl_int),
                                            row_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < row_indices.size(); i++)
                ASSERT_EQ(ublas_coo.index1_data()[i], row_indices[i]);

            // col indices
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.colIndices,
                                            CL_TRUE, 0, col_indices.size() * sizeof(cl_int),
                                            col_indices.data(), 0, nullptr, nullptr);

            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < col_indices.size(); i++)
                ASSERT_EQ(ublas_coo.index2_data()[i], col_indices[i]);


            // values
            cl_status = clEnqueueReadBuffer(CLSE::queue, cooMatrix.values,
                                            CL_TRUE, 0, values.size() * sizeof(T),
                                            values.data(), 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            for (int i = 0; i < values.size(); i++)
                ASSERT_EQ(ublas_coo.value_data()[i], values[i]);


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

typedef ::testing::Types<cl_float, cl_double> TYPES;
//typedef ::testing::Types<cl_float> TYPES;
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
    double alpha;
    double beta;
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
             "Device id within platform.");

    //	Parse the command line options, ignore unrecognized options and collect them into a vector of strings
    //  Googletest itself accepts command line flags that we wish to pass further on
    po::variables_map vm;
    po::parsed_options parsed = po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store( parsed, vm );
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

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, 1, 0,
                                                  CLSE::queue, CLSE::context));
    return RUN_ALL_TESTS();
}
