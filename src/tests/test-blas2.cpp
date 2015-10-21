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

#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>
#include <iomanip>

#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"

//boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/blas.hpp>

// ULP calculation
#include <boost/math/special_functions/next.hpp>

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;
//cl_uint ClSparseEnvironment::N = 1024;
static cl_bool extended_precision = false;
static cl_bool explicit_zeroes = true;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;


template <typename T>
class Blas2 : public ::testing::Test
{

    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

public:


    void SetUp()
    {
        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

        hAlpha = T(CSRE::alpha);
        hBeta = T(CSRE::beta);

        hX = uBLAS::vector<T>(CSRE::n_cols, 1);
        hY = uBLAS::vector<T>(CSRE::n_rows, 2);

        cl_int status;

        gX.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                   hX.size() * sizeof(T), hX.data().begin(),
                                   &status);
        gX.num_values = hX.size();
        ASSERT_EQ(CL_SUCCESS, status);

        gY.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                   hY.size() * sizeof(T), hY.data().begin(),
                                   &status);
        gY.num_values = hY.size();
        ASSERT_EQ(CL_SUCCESS, status);

        gAlpha.value = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR,
                                      sizeof(T), &hAlpha, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        gBeta.value = clCreateBuffer(CLSE::context,
                                     CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     sizeof(T), &hBeta, &status);
        ASSERT_EQ(CL_SUCCESS, status);
    }

    void TearDown()
    {
        ::clReleaseMemObject(gAlpha.value);
        ::clReleaseMemObject(gBeta.value);

        ::clReleaseMemObject(gX.values);
        ::clReleaseMemObject(gY.values);

        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

    }

    // Knuth's Two-Sum algorithm, which allows us to add together two floating
    // point numbers and exactly tranform the answer into a sum and a
    // rounding error.
    // Inputs: x and y, the two inputs to be aded together.
    // In/Out: *sumk_err, which is incremented (by reference) -- holds the
    //         error value as a result of the 2sum calculation.
    // Returns: The non-corrected sum of inputs x and y.
    T two_sum(T x, T y, T *sumk_err)
    {
        // We use this 2Sum algorithm to perform a compensated summation,
        // which can reduce the cummulative rounding errors in our SpMV
        // summation. Our compensated sumation is based on the SumK algorithm
        // (with K==2) from Ogita, Rump, and Oishi, "Accurate Sum and Dot
        // Product" in SIAM J. on Scientific Computing 26(6) pp 1955-1988,
        // Jun. 2005.
        T sumk_s = x + y;
        T bp = sumk_s - x;
        (*sumk_err) += ((x - (sumk_s - bp)) + (y - bp));
        return sumk_s;
    }

    void test_csrmv()
    {
        clsparseStatus status;
        cl_int cl_status;

        clsparseEnableExtendedPrecision(CLSE::control, extended_precision);

        if (typeid(T) == typeid(cl_float) )
        {
            status = clsparseScsrmv(&gAlpha, &CSRE::csrSMatrix, &gX,
                                    &gBeta, &gY, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            float* vals = (float*)&CSRE::ublasSCsr.value_data()[0];
            int* rows = &CSRE::ublasSCsr.index1_data()[0];
            int* cols = &CSRE::ublasSCsr.index2_data()[0];
            for (int row = 0; row < CSRE::n_rows; row++)
            {
                // Summation done at a higher precision to decrease
                // summation errors from rounding.
                hY[row] *= hBeta;
                int row_end = rows[row+1];
                double temp_sum;
                temp_sum = hY[row];
                for (int i = rows[row]; i < rows[row+1]; i++)
                {
                    // Perform: hY[row] += hAlpha * vals[i] * hX[cols[i]];
                    temp_sum += hAlpha * vals[i] * hX[cols[i]];
                }
                hY[row] = temp_sum;
            }

            T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                                       CL_TRUE, CL_MAP_READ,
                                                       0, gY.num_values * sizeof(T),
                                                       0, nullptr, nullptr, &cl_status);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            uint64_t max_ulps = 0;
            uint64_t min_ulps = UINT64_MAX;
            uint64_t total_ulps = 0;
            for (int i = 0; i < hY.size(); i++)
            {
                long long int intDiff = (long long int)boost::math::float_distance(hY[i], host_result[i]);
                intDiff = llabs(intDiff);
                total_ulps += intDiff;
                if (max_ulps < intDiff)
                    max_ulps = intDiff;
                if (min_ulps > intDiff)
                    min_ulps = intDiff;
                // Debug printouts.
                //std::cout << "Row " << i << " Float Ulps: " << intDiff << std::endl;
                //std::cout.precision(9);
                //std::cout << "\tFloat hY[" << i << "] = " << std::scientific << hY[i] << " (0x" << std::hex << *(uint32_t *)&hY[i] << "), " << std::dec;
                //std::cout << "host_result[" << i << "] = " << std::scientific << host_result[i] << " (0x" << std::hex << *(uint32_t *)&host_result[i] << ")" << std::dec << std::endl;
            }
#ifndef NDEBUG
            if (extended_precision)
            {
                std::cout << "Float Min ulps: " << min_ulps << std::endl;
                std::cout << "Float Max ulps: " << max_ulps << std::endl;
                std::cout << "Float Total ulps: " << total_ulps << std::endl;
                std::cout << "Float Average ulps: " << (double)total_ulps/(double)hY.size() <<  " (Size: " << hY.size() << ")" << std::endl;
            }
#endif

            for (int i = 0; i < hY.size(); i++)
            {
                double compare_val = 0.;
                if (extended_precision)
                {
                    // The limit here is somewhat weak because some GPUs don't
                    // support correctly rounded denorms in SPFP mode.
                    if (boost::math::isnormal(hY[i]))
                        compare_val = fabs(hY[i]*1e-3);
                }
                else
                {
                    if (boost::math::isnormal(hY[i]))
                        compare_val = fabs(hY[i]*0.1);
                }
                if (compare_val < 10*FLT_EPSILON)
                    compare_val = 10*FLT_EPSILON;
                ASSERT_NEAR(hY[i], host_result[i], compare_val);
            }

            cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                                  host_result, 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);
        }

        if (typeid(T) == typeid(cl_double) )
        {
            status = clsparseDcsrmv(&gAlpha, &CSRE::csrDMatrix, &gX,
                                    &gBeta, &gY, CLSE::control);

            ASSERT_EQ(clsparseSuccess, status);

            double* vals = (double*)&CSRE::ublasDCsr.value_data()[0];
            int* rows = &CSRE::ublasDCsr.index1_data()[0];
            int* cols = &CSRE::ublasDCsr.index2_data()[0];
            for (int row = 0; row < CSRE::n_rows; row++)
            {
                // Summation done using a compensated summation to decrease
                // summation errors from rounding. This allows us to get
                // smaller errors without requiring quad precision support.
                // This method is like performing summation at quad precision and
                // casting down to double in the end.
                hY[row] *= hBeta;
                int row_end = rows[row+1];
                double temp_sum;
                temp_sum = hY[row];
                T sumk_err = 0.;
                for (int i = rows[row]; i < rows[row+1]; i++)
                {
                    // Perform: hY[row] += hAlpha * vals[i] * hX[cols[i]];
                    temp_sum = two_sum(temp_sum, hAlpha*vals[i]*hX[cols[i]], &sumk_err);
                }
                hY[row] = temp_sum + sumk_err;
            }

            T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                                       CL_TRUE, CL_MAP_READ,
                                                       0, gY.num_values * sizeof(T),
                                                       0, nullptr, nullptr, &cl_status);
            ASSERT_EQ(CL_SUCCESS, cl_status);

            uint64_t max_ulps = 0;
            uint64_t min_ulps = ULLONG_MAX;
            uint64_t total_ulps = 0;
            for (int i = 0; i < hY.size(); i++)
            {
                long long int intDiff = (long long int)boost::math::float_distance(hY[i], host_result[i]);
                intDiff = llabs(intDiff);
                total_ulps += intDiff;
                if (max_ulps < intDiff)
                    max_ulps = intDiff;
                if (min_ulps > intDiff)
                    min_ulps = intDiff;
                // Debug printouts.
                //std::cout << "Row " << i << " Double Ulps: " << intDiff << std::endl;
                //std::cout.precision(17);
                //std::cout << "\tDouble hY[" << i << "] = " << std::scientific << hY[i] << " (0x" << std::hex << *(uint64_t *)&hY[i] << "), " << std::dec;
                //std::cout << "host_result[" << i << "] = " << std::scientific << host_result[i] << " (0x" << std::hex << *(uint64_t *)&host_result[i] << ")" << std::dec << std::endl;
            }
            if (extended_precision)
            {
#ifndef NDEBUG
                std::cout << "Double Min ulps: " << min_ulps << std::endl;
                std::cout << "Double Max ulps: " << max_ulps << std::endl;
                std::cout << "Double Total ulps: " << total_ulps << std::endl;
                std::cout << "Double Average ulps: " << (double)total_ulps/(double)hY.size() <<  " (Size: " << hY.size() << ")" << std::endl;
#endif

                for (int i = 0; i < hY.size(); i++)
                {
                    double compare_val = fabs(hY[i]*1e-14);
                    if (compare_val < 10*DBL_EPSILON)
                        compare_val = 10*DBL_EPSILON;
                    ASSERT_NEAR(hY[i], host_result[i], compare_val);
                }
            }
            else
            {
                for (int i = 0; i < hY.size(); i++)
                {
                    double compare_val = 0.;
                    if (boost::math::isnormal(hY[i]))
                        compare_val = fabs(hY[i]*0.1);
                    if (compare_val < 10*DBL_EPSILON)
                        compare_val = 10*DBL_EPSILON;
                    ASSERT_NEAR(hY[i], host_result[i], compare_val);
                }
            }

            cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                                  host_result, 0, nullptr, nullptr);
            ASSERT_EQ(CL_SUCCESS, cl_status);
        }
        // Reset output buffer for next test.
        ::clReleaseMemObject(gY.values);
        clsparseInitVector(&gY);
        gY.values = clCreateBuffer(CLSE::context,
                CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                hY.size() * sizeof(T), hY.data().begin(),
                &cl_status);
        gY.num_values = hY.size();
        ASSERT_EQ(CL_SUCCESS, cl_status);
    }

    uBLAS::vector<T> hX;
    uBLAS::vector<T> hY;

    cldenseVector gX;
    cldenseVector gY;

    T hAlpha;
    T hBeta;

    clsparseScalar gAlpha;
    clsparseScalar gBeta;

};

// CSRMV Tests might give you a false ASSERT_NEAR depends on the matrix.
// We should calculate relative error instead of ASSERT_NEAR.

typedef ::testing::Types<cl_float, cl_double> TYPES;
TYPED_TEST_CASE(Blas2, TYPES);

TYPED_TEST(Blas2, csrmv_adaptive)
{
    this->test_csrmv();
}

TYPED_TEST(Blas2, csrmv_vector)
{
    // To call csrmv vector we need to artificially get rid of the rowBlocks data
    using CSRE = CSREnvironment;

    cl_int cl_status;
    cl_status = clReleaseMemObject(CSRE::csrSMatrix.rowBlocks);
    ASSERT_EQ(CL_SUCCESS, cl_status);
    CSRE::csrSMatrix.rowBlocks = nullptr;

    cl_status = clReleaseMemObject(CSRE::csrDMatrix.rowBlocks);
    ASSERT_EQ(CL_SUCCESS, cl_status);
    CSRE::csrDMatrix.rowBlocks = nullptr;

    CSRE::csrSMatrix.rowBlockSize = 0;
    CSRE::csrDMatrix.rowBlockSize = 0;

    this->test_csrmv();

    // After calling the kernel we need to recreate the rowBlocks data for
    // later use.

    clsparseStatus status;
    status = clsparseCsrMetaSize( &CSRE::csrSMatrix, CLSE::control );
    ASSERT_EQ(clsparseSuccess, status);

    status = clsparseCsrMetaSize( &CSRE::csrDMatrix, CLSE::control );
    ASSERT_EQ(clsparseSuccess, status);

    CSRE::csrSMatrix.rowBlocks =
            ::clCreateBuffer( CLSE::context, CL_MEM_READ_WRITE,
                              CSRE::csrSMatrix.rowBlockSize * sizeof( cl_ulong ),
                              NULL, &cl_status );

    ASSERT_EQ(CL_SUCCESS, cl_status);

    CSRE::csrDMatrix.rowBlocks = CSRE::csrSMatrix.rowBlocks;
    ::clRetainMemObject( CSRE::csrDMatrix.rowBlocks );

    status = clsparseCsrMetaCompute(&CSRE::csrSMatrix, CLSE::control );
    ASSERT_EQ (clsparseSuccess, status);
    status = clsparseCsrMetaCompute(&CSRE::csrDMatrix, CLSE::control );
    ASSERT_EQ (clsparseSuccess, status);
}


int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

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
             "Device id within platform.")
            ("alpha,a", po::value(&alpha)->default_value(1.0),
             "Alpha parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(1.0),
             "Beta parameter for eq: \n\ty = alpha * M * x + beta * y")
            ("extended,e", po::bool_switch()->default_value(false),
             "Use compensated summation to improve accuracy by emulating extended precision.")
            ("no_zeroes,z", po::bool_switch()->default_value(false),
             "Disable reading explicit zeroes from the input matrix market file.");

    po::variables_map vm;
    po::parsed_options parsed =
            po::command_line_parser( argc, argv ).options( desc ).allow_unregistered( ).run( );

    try {
        po::store(parsed, vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    std::vector< std::string > to_pass_further =
            po::collect_unrecognized( parsed.options, po::include_positional );

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

    if (vm["extended"].as<bool>())
        extended_precision = true;
    if (vm["no_zeroes"].as<bool>())
        explicit_zeroes = false;

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID));
    ::testing::AddGlobalTestEnvironment( new CSRE(path, alpha, beta,
                                                  CLSE::queue, CLSE::context, explicit_zeroes));

    return RUN_ALL_TESTS();

}
