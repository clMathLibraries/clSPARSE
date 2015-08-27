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
#include <numeric>
#include <boost/program_options.hpp>

//boost ublas
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/blas.hpp>

#include "resources/clsparse_environment.h"
#include "resources/blas1_environment.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

double Blas1Environment::alpha = 1;
double Blas1Environment::beta = 1;
int Blas1Environment::size = 1024;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;

template <typename T>
class ScanTest : public ::testing::Test
{
    using CLSE = ClSparseEnvironment;
    using BLAS1E = Blas1Environment;

public:

    std::vector<T> hX;
    std::vector<T> hY;

    cldenseVector gX;
    cldenseVector gY;

    T hAlpha = BLAS1E::alpha;
    T hBeta = BLAS1E::beta;




    void SetUp()
    {
        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

        hX = std::vector<T>(BLAS1E::size, hAlpha);
        hY = std::vector<T>(BLAS1E::size, hBeta);

        std::iota(hY.begin(), hY.end(), -3);

        cl_int status;
        gX.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hX.size() * sizeof(T), hX.data(), &status);
        gX.num_values = hX.size();
        ASSERT_EQ(CL_SUCCESS, status);


        gY.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hY.size() * sizeof(T), hY.data(), &status);

        gY.num_values = hY.size();
        ASSERT_EQ(CL_SUCCESS, status);


    }

    void TearDown()
    {
        clReleaseMemObject( gX.values );
        clReleaseMemObject( gY.values );
    }

    void test_exclusive_scan()
    {
        clsparseStatus status;

        if (typeid(T) == typeid(cl_int))
        {
            status = clsparseIscan(&gX, &gY, true,  CLSE::control);
        }

        else if (typeid(T) == typeid(cl_float))
        {
            status = clsparseSscan(&gX, &gY, true,  CLSE::control);
        }

        else if (typeid(T) == typeid(cl_double))
        {
            status = clsparseDscan(&gX, &gY, true,  CLSE::control);
        }
        else
        {
            status = clsparseInvalidOperation;
        }

        ASSERT_EQ(clsparseSuccess, status);

        host_scan(hX, hY, true);

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gX.values,
                                                   CL_TRUE, CL_MAP_READ,
                                                   0, gX.num_values * sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        if ( typeid(T) == typeid(cl_int) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "int h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_EQ (hX[i], host_result[i]);
            }

        if ( typeid(T) == typeid(cl_float) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "float h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_NEAR (hX[i], host_result[i], 1e-7);
            }

        if ( typeid(T) == typeid(cl_double) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "double h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_NEAR (hX[i], host_result[i], 1e-14);
            }

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gX.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

    void test_inclusive_scan()
    {
        clsparseStatus status;


        if (typeid(T) == typeid(cl_int))
        {
            status = clsparseIscan(&gX, &gY, false, CLSE::control);
        }

        else if (typeid(T) == typeid(cl_float))
        {
            status = clsparseSscan(&gX, &gY, false, CLSE::control);
        }

        else if (typeid(T) == typeid(cl_double))
        {
            status = clsparseDscan(&gX, &gY, false, CLSE::control);
        }
        else
        {
            status = clsparseInvalidOperation;
        }

        ASSERT_EQ(clsparseSuccess, status);

        host_scan(hX, hY, false);

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gX.values,
                                                   CL_TRUE, CL_MAP_READ,
                                                   0, gX.num_values * sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        if ( typeid(T) == typeid(cl_int) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "int h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_EQ (hX[i], host_result[i]);
            }

        if ( typeid(T) == typeid(cl_float) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "float h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_NEAR (hX[i], host_result[i], 1e-7);
            }

        if ( typeid(T) == typeid(cl_double) )
            for (int i = 0; i < hX.size(); i++)
            {
                //std::cout << "double h[x] = " << hX[i] << " res = " << host_result[i] << std::endl;
                ASSERT_NEAR (hX[i], host_result[i], 1e-14);
            }

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gX.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }
    void host_scan(std::vector<T>& output,
                   const std::vector<T>& input,
                   bool exclusive)
    {
        if (exclusive)
            exclusive_scan(output, input);
        else
            inclusive_scan(output, input);
    }



private:
    void inclusive_scan(std::vector<T>& output,
                        const std::vector<T>& input)
    {
        T accumulator = 0;

        for (int i = 0; i < input.size(); i++)
        {

            accumulator = accumulator + input[i];

            output[i] = accumulator;
        }

    }

    void exclusive_scan(std::vector<T>& output,
                        const std::vector<T>& input)
    {
        T accumulator = 0;

        for (int i = 0; i < input.size(); i++)
        {
            output[i] = accumulator;

            accumulator = accumulator + input[i];
        }
    }

};


typedef ::testing::Types<cl_float, cl_double, cl_int> TYPES;

TYPED_TEST_CASE(ScanTest, TYPES);

TYPED_TEST(ScanTest, inclusive_scan)
{
    this->test_inclusive_scan();
}

TYPED_TEST(ScanTest, exclusive_scan)
{
    this->test_exclusive_scan();
}


int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;
    using BLAS1E = Blas1Environment;


    std::string platform;
    cl_platform_type pID;
    cl_uint dID;
    cl_uint size;

    double alpha;
    double beta;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.")
            ("size,s",po::value(&size)->default_value(1024),
             "Size of the vectors used for testing")
            ("alpha,a", po::value(&alpha)->default_value(2),
             "Alpha coefficient for blas1 operations i.e. r = alpha * x + beta * y")
            ("beta,b", po::value(&beta)->default_value(4),
             "Beta coefficient for blas1 operations i.e. r = alpha * x + beta * y");


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

    ::testing::InitGoogleTest(&argc, argv);
    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE(pID, dID, size));
    ::testing::AddGlobalTestEnvironment( new BLAS1E(alpha, beta, size));

    return RUN_ALL_TESTS();

}
