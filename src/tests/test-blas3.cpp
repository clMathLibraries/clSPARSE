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

#if defined ( _WIN32 )
#define NOMINMAX
#endif

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/sparse_matrix_environment.h"
#include "resources/sparse_matrix_fill.hpp"
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

//#define _DEBUG_SpMxSpM_ 1 // For debugging where errors are occuring
const float SPGEMM_PREC_ERROR = 0.2f;
const float SPGEMM_REL_ERROR = 0.001f;

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

static cl_bool explicit_zeroes = true;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;

//number of columns in dense B matrix;
cl_int B_num_cols;
cl_double B_values;

template<typename T>
clsparseStatus generateSpGemmResult(clsparseCsrMatrix& sparseMatC)
{
    using SPER = CSRSparseEnvironment;
    using CLSE = ClSparseEnvironment;

    if (typeid(T) == typeid(float))
    {
        return clsparseScsrSpGemm(&SPER::csrSMatrix, &SPER::csrSMatrix, &sparseMatC, CLSE::control);
    }
    /*
    else if (typeid(T) == typeid(double))
    {
        return clsparseDcsrSpGemm(SPER::csrSMatrix, SPER::csrSMatrix, sparseMatC, CLSE::control);
    }*/

    return clsparseSuccess;
}// end

#ifdef TEST_LONG
    template<typename T>
    clsparseStatus generateSpGemmResult_long(clsparseCsrMatrix& sparseMatC)
    {
       using SPER = CSRSparseEnvironment;
       using CLSE = ClSparseEnvironment;

       if (typeid(T) == typeid(float))
       {
          return clsparseScsrSpGemm(&SPER::csrSMatrixA, &SPER::csrSMatrixB, &sparseMatC, CLSE::control);
       }
    /*
    else if (typeid(T) == typeid(double))
    {
        return clsparseDcsrSpGemm(SPER::csrSMatrix, SPER::csrSMatrix, sparseMatC, CLSE::control);
    }*/

    return clsparseSuccess;
}// end

#endif

template <typename T>
class TestCSRSpGeMM : public ::testing::Test {

    using SPER = CSRSparseEnvironment;
    using CLSE = ClSparseEnvironment;

public:
    void SetUp()
    {
        clsparseInitCsrMatrix(&csrMatrixC);
    }

    void TearDown()
    {
        ::clReleaseMemObject(csrMatrixC.values);
        ::clReleaseMemObject(csrMatrixC.colIndices);
        ::clReleaseMemObject(csrMatrixC.rowOffsets);

        clsparseInitCsrMatrix(&csrMatrixC);
    }// end
    
    void checkRowOffsets(std::vector<clsparseIdx_t>& amdRowPtr)
    {
        for (clsparseIdx_t i = 0; i < amdRowPtr.size(); i++)
        {
            //ASSERT_EQ(amdRowPtr[i], this->C.index1_data()[i]);
            //EXPECT_EQ(amdRowPtr[i], this->C.index1_data()[i]);
            if (amdRowPtr[i] != this->C.index1_data()[i])
            {
                this->browOffsetsMisFlag = true;
                break;
            }
        }
    }// end

    void checkInDense(std::vector<clsparseIdx_t>& amdRowPtr, std::vector<clsparseIdx_t>& amdColIndices, std::vector<T>& amdVals)
    {
        uBLAS::mapped_matrix<T> sparseDense(csrMatrixC.num_rows, csrMatrixC.num_cols, 0);
        uBLAS::mapped_matrix<T> boostDense(csrMatrixC.num_rows, csrMatrixC.num_cols, 0);

        // boost sparse_prod cancels out zeros and hence reports more accurately non-zeros
        // In clSPARSE, spGeMM produces more non-zeros, and considers some zeros as nonzeros.
        // Therefore converting to dense and verifying the output in  dense format
        // Convert CSR to Dense

        for (clsparseIdx_t i = 0; i < amdRowPtr.size() - 1; i++)
        {
            // i corresponds to row index
            for (clsparseIdx_t j = amdRowPtr[i]; j < amdRowPtr[i + 1]; j++)
                sparseDense(i, amdColIndices[j]) = amdVals[j];
        }

        for (clsparseIdx_t i = 0; i < this->C.index1_data().size() - 1; i++)
        {
            for (clsparseIdx_t j = this->C.index1_data()[i]; j < this->C.index1_data()[i + 1]; j++)
                boostDense(i, this->C.index2_data()[j]) = this->C.value_data()[j];
        }

        bool brelativeErrorFlag = false;
        bool babsErrorFlag = false;
        
        for (clsparseIdx_t i = 0; i < csrMatrixC.num_rows; i++)
        {
            for (clsparseIdx_t j = 0; j < csrMatrixC.num_cols; j++)
            {
                //ASSERT_EQ(boostDense(i, j), sparseDense(i, j));
#ifdef _DEBUG_SpMxSpM_
                ASSERT_NEAR(boostDense(i, j), sparseDense(i, j), SPGEMM_PREC_ERROR);
#else
                if (fabs(boostDense(i, j) - sparseDense(i, j)) > SPGEMM_PREC_ERROR)
                {
                    babsErrorFlag = true;
                    SCOPED_TRACE("Absolute Error Fail");
                    break;
                }
#endif
            }
        }
        // Relative Error
        for (clsparseIdx_t i = 0; i < csrMatrixC.num_rows; i++)
        {
            for (clsparseIdx_t j = 0; j < csrMatrixC.num_cols; j++)
            {
                float diff  = fabs(boostDense(i, j) - sparseDense(i, j));
                float ratio = diff / boostDense(i, j);
#ifdef _DEBUG_SpMxSpM_
                // ratio is less than or almost equal to SPGEMM_REL_ERROR
                EXPECT_PRED_FORMAT2(::testing::FloatLE, ratio, SPGEMM_REL_ERROR);
#else
                if (diff / boostDense(i, j) > SPGEMM_REL_ERROR)
                {
                    brelativeErrorFlag = true;
                    SCOPED_TRACE("Relative Error Fail");
                    break;
                }
#endif
            }
        }//
#ifndef _DEBUG_SpMxSpM_
        if (brelativeErrorFlag)
        {
            ASSERT_FALSE(babsErrorFlag);
        }

        if (babsErrorFlag)
        {
            ASSERT_FALSE(brelativeErrorFlag);
        }
#endif
    }// end

    typedef typename uBLAS::compressed_matrix<T, uBLAS::row_major, 0, uBLAS::unbounded_array<size_t> > uBlasCSRM;

    uBlasCSRM C;
    bool browOffsetsMisFlag;
    clsparseCsrMatrix csrMatrixC;
}; // End of class TestCSRSpGeMM

//typedef ::testing::Types<float, double> SPGEMMTYPES;
typedef ::testing::Types<float> SPGEMMTYPES;
TYPED_TEST_CASE(TestCSRSpGeMM, SPGEMMTYPES);

// C = A * A; // Square matrices are only supported
TYPED_TEST(TestCSRSpGeMM, square)
{
    using SPER = CSRSparseEnvironment;
    using CLSE = ClSparseEnvironment;
    typedef typename uBLAS::compressed_matrix<float, uBLAS::row_major, 0, uBLAS::unbounded_array<clsparseIdx_t> > uBlasCSRM;
 
    cl::Event event;
    clsparseEnableAsync(CLSE::control, true);

#ifdef TEST_LONG
    clsparseStatus status = generateSpGemmResult_long<TypeParam>(this->csrMatrixC);
#else
    clsparseStatus status = generateSpGemmResult<TypeParam>(this->csrMatrixC);
#endif

    EXPECT_EQ(clsparseSuccess, status);

    status = clsparseGetEvent(CLSE::control, &event());
    EXPECT_EQ(clsparseSuccess, status);
    event.wait();

    //std::cout << "nrows =" << (this->csrMatrixC).num_rows << std::endl;
    //std::cout << "nnz =" << (this->csrMatrixC).num_nonzeros << std::endl;

    std::vector<clsparseIdx_t> resultRowPtr((this->csrMatrixC).num_rows + 1); // Get row ptr of Output CSR matrix
    std::vector<clsparseIdx_t> resultColIndices((this->csrMatrixC).num_nonzeros); // Col Indices
    std::vector<TypeParam> resultVals((this->csrMatrixC).num_nonzeros); // Values

    this->C = uBlasCSRM((this->csrMatrixC).num_rows, (this->csrMatrixC).num_cols, (this->csrMatrixC).num_nonzeros);
    (this->C).complete_index1_data();

    cl_int cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.values, CL_TRUE, 0,
        (this->csrMatrixC).num_nonzeros *sizeof(TypeParam),
        resultVals.data(), 0, NULL, NULL);
    
    EXPECT_EQ(CL_SUCCESS, cl_status);

    
    cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.colIndices, CL_TRUE, 0,
        (this->csrMatrixC).num_nonzeros * sizeof(clsparseIdx_t), resultColIndices.data(), 0, NULL, NULL);
    
    EXPECT_EQ(CL_SUCCESS, cl_status);

    
    cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.rowOffsets, CL_TRUE, 0,
        ((this->csrMatrixC).num_rows + 1)  * sizeof(clsparseIdx_t), resultRowPtr.data(), 0, NULL, NULL);

    EXPECT_EQ(CL_SUCCESS, cl_status);

    std::cout << "Done with GPU" << std::endl;

#ifdef TEST_LONG 
    // Generate referencee result from ublas
    if (typeid(TypeParam) == typeid(float))
    {
        this->C = uBLAS::sparse_prod(SPER::ublasSCsrA, SPER::ublasSCsrB, this->C);
    }
#else
    if (typeid(TypeParam) == typeid(float))
    {
        this->C = uBLAS::sparse_prod(SPER::ublasSCsr, SPER::ublasSCsr, this->C);
    }

#endif
    
    /*
    if (typeid(TypeParam) == typeid(double))
    {
        this->C = uBLAS::sparse_prod(SPER::ublasDCsr, SPER::ublasDCsr, this->C);;
    }*/

    /*
    for (int i = 0; i < resultRowPtr.size(); i++)
    {
        ASSERT_EQ(resultRowPtr[i], this->C.index1_data()[i]);
    }*/
    this->browOffsetsMisFlag = false;
   this->checkRowOffsets(resultRowPtr);
   //if (::testing::Test::HasFailure())
   if (this->browOffsetsMisFlag == true)
    {
        // Check the values in Dense format
        this->checkInDense(resultRowPtr, resultColIndices, resultVals);
    }
    else
    {
        /* Check Col Indices */
        for (clsparseIdx_t i = 0; i < resultColIndices.size(); i++)
        {
            ASSERT_EQ(resultColIndices[i], this->C.index2_data()[i]);
        }

        /* Check Values */
        for (clsparseIdx_t i = 0; i < resultVals.size(); i++)
        {
            //TODO: how to define the tolerance 
            ASSERT_NEAR(resultVals[i], this->C.value_data()[i], 0.1);
        }

        ASSERT_EQ(resultRowPtr.size(), this->C.index1_data().size());

        //Rest of the col_indices should be zero
        for (size_t i = resultColIndices.size(); i < this->C.index2_data().size(); i++)
        {
            ASSERT_EQ(0, this->C.index2_data()[i]);
        }

        // Rest of the values should be zero
        for (size_t i = resultVals.size(); i < this->C.value_data().size(); i++)
        {
            ASSERT_EQ(0, this->C.value_data()[i]);
        }
    }

}//end TestCSRSpGeMM: square


// C = A * A; // A is filled with random powers of 2
TYPED_TEST(TestCSRSpGeMM, Powersof2)
{
    using SPER = CSRSparseEnvironment;
    using CLSE = ClSparseEnvironment;
    typedef typename uBLAS::compressed_matrix<float, uBLAS::row_major, 0, uBLAS::unbounded_array<clsparseIdx_t> > uBlasCSRM;

    cl::Event event;
    clsparseEnableAsync(CLSE::control, true);

    clsparse_matrix_fill<float> objFillVals(42, -14, 14);

    std::vector<float> tmpArray;
    tmpArray.resize(SPER::csrSMatrix.num_nonzeros);

    //objFillVals.fillMtxTwoPowers(tmpArray.data(), tmpArray.size());
    objFillVals.fillMtxOnes(tmpArray.data(), tmpArray.size());

    // Fill ublas scr with the same matrix values
    for (size_t i = 0; i < tmpArray.size(); i++)
    {
        SPER::ublasSCsr.value_data()[i] = tmpArray[i];
    }
    
    // Copy host to the device
    cl_int cl_status = clEnqueueWriteBuffer(CLSE::queue, SPER::csrSMatrix.values, CL_TRUE, 0, sizeof(float)* tmpArray.size(),
                                                 tmpArray.data(), 0, nullptr, nullptr);
    EXPECT_EQ(CL_SUCCESS, cl_status);
    tmpArray.clear();

    clsparseStatus status = generateSpGemmResult<TypeParam>(this->csrMatrixC);

    EXPECT_EQ(clsparseSuccess, status);

    status = clsparseGetEvent(CLSE::control, &event());
    EXPECT_EQ(clsparseSuccess, status);
    event.wait();


    std::vector<clsparseIdx_t> resultRowPtr((this->csrMatrixC).num_rows + 1); // Get row ptr of Output CSR matrix
    std::vector<clsparseIdx_t> resultColIndices((this->csrMatrixC).num_nonzeros); // Col Indices
    std::vector<TypeParam> resultVals((this->csrMatrixC).num_nonzeros); // Values

    this->C = uBlasCSRM((this->csrMatrixC).num_rows, (this->csrMatrixC).num_cols, (this->csrMatrixC).num_nonzeros);
    (this->C).complete_index1_data();

    cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.values, CL_TRUE, 0,
        (this->csrMatrixC).num_nonzeros *sizeof(TypeParam),
        resultVals.data(), 0, NULL, NULL);

    EXPECT_EQ(CL_SUCCESS, cl_status);


    cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.colIndices, CL_TRUE, 0,
        (this->csrMatrixC).num_nonzeros * sizeof(clsparseIdx_t), resultColIndices.data(), 0, NULL, NULL);

    EXPECT_EQ(CL_SUCCESS, cl_status);


    cl_status = clEnqueueReadBuffer(CLSE::queue,
        this->csrMatrixC.rowOffsets, CL_TRUE, 0,
        ((this->csrMatrixC).num_rows + 1)  * sizeof(clsparseIdx_t), resultRowPtr.data(), 0, NULL, NULL);

    EXPECT_EQ(CL_SUCCESS, cl_status);

    std::cout << "Done with GPU" << std::endl;

    if (typeid(TypeParam) == typeid(float))
    {
        this->C = uBLAS::sparse_prod(SPER::ublasSCsr, SPER::ublasSCsr, this->C);
    }

    this->browOffsetsMisFlag = false;
    this->checkRowOffsets(resultRowPtr);
    //if (::testing::Test::HasFailure())
    if (this->browOffsetsMisFlag == true)
    {
        // Check the values in Dense format
        this->checkInDense(resultRowPtr, resultColIndices, resultVals);
    }
    else
    {
        /* Check Col Indices */
        for (clsparseIdx_t i = 0; i < resultColIndices.size(); i++)
        {
            ASSERT_EQ(resultColIndices[i], this->C.index2_data()[i]);
        }

        /* Check Values */
        for (clsparseIdx_t i = 0; i < resultVals.size(); i++)
        {
            //TODO: how to define the tolerance 
            ASSERT_NEAR(resultVals[i], this->C.value_data()[i], 0.0);
        }

        ASSERT_EQ(resultRowPtr.size(), this->C.index1_data().size());

        //Rest of the col_indices should be zero
        for (clsparseIdx_t i = resultColIndices.size(); i < this->C.index2_data().size(); i++)
        {
            ASSERT_EQ(0, this->C.index2_data()[i]);
        }

        // Rest of the values should be zero
        for (clsparseIdx_t i = resultVals.size(); i < this->C.value_data().size(); i++)
        {
            ASSERT_EQ(0, this->C.value_data()[i]);
        }
    }

}//end TestCSRSpGeMM: Powersof2






template<typename T>
clsparseStatus generateResult( cldenseMatrix& matB, clsparseScalar& alpha,
                               cldenseMatrix& matC, clsparseScalar& beta )
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    if(typeid(T) == typeid(float))
    {

        return clsparseScsrmm( &alpha, &CSRE::csrSMatrix, &matB,
                               &beta, &matC, CLSE::control );


    }

    if(typeid(T) == typeid(double))
    {
        return clsparseDcsrmm( &alpha, &CSRE::csrDMatrix, &matB,
                               &beta, &matC, CLSE::control );

    }
    return clsparseSuccess;
}

template <typename T>
class TestCSRMM : public ::testing::Test
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

public:

    void SetUp()
    {
        // alpha and beta scalars are not yet supported in generating reference result;
        alpha = T(CSRE::alpha);
        beta = T(CSRE::beta);

        B = uBLASDenseM(CSRE::n_cols, B_num_cols, T(B_values));
        C = uBLASDenseM(CSRE::n_rows, B_num_cols, T(0));


        cl_int status;
        cldenseInitMatrix( &deviceMatB );
        deviceMatB.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   B.data().size( ) * sizeof( T ), B.data().begin(), &status );

        deviceMatB.num_rows = B.size1();
        deviceMatB.num_cols = B.size2();
        deviceMatB.lead_dim = std::min(B.size1(), B.size2());


        ASSERT_EQ(CL_SUCCESS, status);

        cldenseInitMatrix( &deviceMatC );
        deviceMatC.values = clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   C.data().size( ) * sizeof( T ), C.data().begin(), &status );


        deviceMatC.num_rows = C.size1();
        deviceMatC.num_cols = C.size2();
        deviceMatC.lead_dim = std::min(C.size1(), C.size2());
        ASSERT_EQ(CL_SUCCESS, status);

        clsparseInitScalar( &gAlpha );
        gAlpha.value = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(T), &alpha, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        clsparseInitScalar( &gBeta );
        gBeta.value = clCreateBuffer(CLSE::context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(T), &beta, &status);
        ASSERT_EQ(CL_SUCCESS, status);

    }

    void TearDown()
    {
        ::clReleaseMemObject(gAlpha.value);
        ::clReleaseMemObject(gBeta.value);

        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        ::clReleaseMemObject(deviceMatB.values);
        ::clReleaseMemObject(deviceMatC.values);

        cldenseInitMatrix( &deviceMatB );
        cldenseInitMatrix( &deviceMatC );

    }


    typedef typename uBLAS::matrix<T, uBLAS::row_major, uBLAS::unbounded_array<T> > uBLASDenseM;
    uBLASDenseM B;
    uBLASDenseM C;


    cldenseMatrix deviceMatB;
    cldenseMatrix deviceMatC;

    T alpha;
    T beta;

    clsparseScalar gAlpha;
    clsparseScalar gBeta;
};

typedef ::testing::Types<float,double> TYPES;
//typedef ::testing::Types<float> TYPES;
TYPED_TEST_CASE( TestCSRMM, TYPES );



// This test may give you false failure result due to multiplication order.
TYPED_TEST(TestCSRMM, multiply)
{
    using CSRE = CSREnvironment;
    using CLSE = ClSparseEnvironment;

    cl::Event event;
    clsparseEnableAsync(CLSE::control, true);

    //control object is global and it is updated here;
    clsparseStatus status =
            generateResult<TypeParam>(this->deviceMatB, this->gAlpha,
            this->deviceMatC, this->gBeta );

    EXPECT_EQ(clsparseSuccess, status);

    status = clsparseGetEvent(CLSE::control, &event());
    EXPECT_EQ(clsparseSuccess, status);
    event.wait();

    std::vector<TypeParam> result(this->C.data().size());

    cl_int cl_status = clEnqueueReadBuffer(CLSE::queue,
                                            this->deviceMatC.values, CL_TRUE, 0,
                                            result.size()*sizeof(TypeParam),
                                            result.data(), 0, NULL, NULL);
    EXPECT_EQ(CL_SUCCESS, cl_status);

    // Generate referencee result;
    if (typeid(TypeParam) == typeid(float))
    {
         this->C = uBLAS::sparse_prod(CSRE::ublasSCsr, this->B, this->C, false);
    }

    if (typeid(TypeParam) == typeid(double))
    {
         this->C = uBLAS::sparse_prod(CSRE::ublasDCsr, this->B, this->C, false);
    }


    if(typeid(TypeParam) == typeid(float))
        for (clsparseIdx_t l = 0; l < std::min(this->C.size1(), this->C.size2()); l++)
            for (clsparseIdx_t i = 0; i < this->C.data().size(); i++)
            {
                ASSERT_NEAR(this->C.data()[i], result[i], 5e-3);
            }

    if(typeid(TypeParam) == typeid(double))
        for (clsparseIdx_t l = 0; l < std::min(this->C.size1(), this->C.size2()); l++)
            for (clsparseIdx_t i = 0; i < this->C.data().size(); i++)
            {
                ASSERT_NEAR(this->C.data()[i], result[i], 5e-10);
            };
}


int main (int argc, char* argv[])
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;
    using SPER = CSRSparseEnvironment;
    //pass path to matrix as an argument, We can switch to boost po later

    std::string path;
    std::string function;
    double alpha;
    double beta;
    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("path,p", po::value(&path)->required(), "Path to matrix in mtx format.")
            ("function,f", po::value<std::string>(&function)->default_value("SpMdM"), "Sparse functions to test. Options: SpMdM, SpMSpM, All")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(0.0),
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("cols,c", po::value(&B_num_cols)->default_value(8),
             "Number of columns in B matrix while calculating sp_A * d_B = d_C")
            ("vals,v", po::value(&B_values)->default_value(1.0),
             "Initial value of B columns")
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

    if (boost::iequals(function, "SpMdM"))
    {
        std::cout << "SpMdM Testing \n";
        ::testing::GTEST_FLAG(filter) = "*TestCSRMM*" ;
        //::testing::GTEST_FLAG(list_tests) = true;

        ::testing::InitGoogleTest(&argc, argv);
        //order does matter!
        ::testing::AddGlobalTestEnvironment(new CLSE(pID, dID));
        ::testing::AddGlobalTestEnvironment(new CSRE(path, alpha, beta,
            CLSE::queue, CLSE::context, explicit_zeroes));

    }
    else if (boost::iequals(function, "SpMSpM"))
    {
        std::cout << "SpMSpM Testing \n";
        ::testing::GTEST_FLAG(filter) = "*TestCSRSpGeMM*" ;
        //::testing::GTEST_FLAG(list_tests) = true;
        
        ::testing::InitGoogleTest(&argc, argv);

        ::testing::AddGlobalTestEnvironment(new CLSE(pID, dID));
        ::testing::AddGlobalTestEnvironment(new SPER(path, CLSE::queue, CLSE::context, explicit_zeroes));
    }
    else if (boost::iequals(function, "All"))
    {
        ::testing::InitGoogleTest(&argc, argv);
        //order does matter!
        ::testing::AddGlobalTestEnvironment(new CLSE(pID, dID));
        ::testing::AddGlobalTestEnvironment(new CSRE(path, alpha, beta,
            CLSE::queue, CLSE::context, explicit_zeroes));
        ::testing::AddGlobalTestEnvironment(new SPER(path, CLSE::queue, CLSE::context, explicit_zeroes));
    }
    else
    {
        std::cerr << " unknown Level3 function" << std::endl;
        return -1;
    }
    
    return RUN_ALL_TESTS();
}// end
