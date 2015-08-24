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
#include <algorithm>
#include <random>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"
#include "resources/blas1_environment.h"


clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

double Blas1Environment::alpha = 1;
double Blas1Environment::beta = 1;
int Blas1Environment::size = 1024;

namespace po = boost::program_options;


template<typename T>
class ReduceByKey : public ::testing::Test
{
    using CLSE = ClSparseEnvironment;
    using BLAS1E = Blas1Environment;

public:

    typedef int KeyType;

    std::vector<KeyType> hKeys;
    std::vector<T> hValues;

    cldenseVector gKeys;
    cldenseVector gValues;

    T hAlpha = BLAS1E::alpha;
    T hBeta = BLAS1E::beta;

    void SetUp()
    {
        clsparseInitVector(&gKeys);
        clsparseInitVector(&gValues);

        hKeys = std::vector<KeyType>(BLAS1E::size);
        hValues = std::vector<T>(BLAS1E::size);

        randomize_keys();

        cl_int status;
        gKeys.values = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      hKeys.size() * sizeof(KeyType), hKeys.data(), &status);
        gKeys.num_values = hKeys.size();
        ASSERT_EQ(CL_SUCCESS, status);


        gValues.values = clCreateBuffer(CLSE::context,
                                        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                        hValues.size() * sizeof(T), hValues.data(), &status);

        gValues.num_values = hValues.size();
        ASSERT_EQ(CL_SUCCESS, status);

    }

    void TearDown()
    {
        clReleaseMemObject( gKeys.values );
        clReleaseMemObject( gValues.values );
    }


    template<typename K, typename V>
    void host_reduce_by_key(std::vector<K>& kOut, std::vector<V>& vOut,
                            const std::vector<K>& kIn, const std::vector<V>& vIn)
    {
        std::unique_copy(kIn.begin(), kIn.end(), std::back_inserter(kOut));

        vOut.resize(kOut.size());

        int j = 0;
        for (int k = 0; k < kIn.size(); k++)
        {
            if (kOut[j] == kIn[k])
                vOut[j] += vIn[k];
            else
            {
                j++;
                vOut[j] += vIn[k];
            }
        }
    }

    void test_reduce_by_key()
    {
        std::vector<KeyType> hKeysOut(0);
        std::vector<T> hValsOut(0);

        host_reduce_by_key(hKeysOut, hValsOut, hKeys, hValues);

        auto reduced_size = hKeysOut.size();

        cldenseVector gKeysOut;
        cldenseVector gValsOut;

        clsparseInitVector(&gKeysOut);
        clsparseInitVector(&gValsOut);

        cl_int status;

        gKeysOut.values = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                         hKeysOut.size() * sizeof(KeyType),
                                         nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);
        gKeysOut.num_values = hKeysOut.size();

        gValsOut.values = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                         hValsOut.size() * sizeof(T),
                                         nullptr, &status);

        ASSERT_EQ(CL_SUCCESS, status);
        gValsOut.num_values = hValsOut.size();


        if (typeid(cl_float) == typeid(T))
        {
            clsparseSreduceByKey(&gKeysOut, &gValsOut, &gKeys, &gValues, CLSE::control);
        }

        if (typeid(cl_double) == typeid(T))
        {
            clsparseDreduceByKey(&gKeysOut, &gValsOut, &gKeys, &gValues, CLSE::control);
        }

        T* host_values = (T*) clEnqueueMapBuffer(CLSE::queue, gValsOut.values,
                                                 CL_TRUE, CL_MAP_READ,
                                                 0, gValsOut.num_values * sizeof(T),
                                                 0, nullptr, nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);

        KeyType* host_keys = (KeyType*) clEnqueueMapBuffer(CLSE::queue, gKeysOut.values,
                                                           CL_TRUE, CL_MAP_READ,
                                                           0, gKeysOut.num_values * sizeof(KeyType),
                                                           0, nullptr, nullptr, &status);
        ASSERT_EQ(CL_SUCCESS, status);


        // check the results;
        if(typeid(cl_float) == typeid(T))
        {
            for (int i = 0; i < hKeysOut.size(); i++)
            {
                ASSERT_EQ(hKeysOut[i], host_keys[i]);
                ASSERT_NEAR(hValsOut[i], host_values[i], 1e-7);
            }
        }

        if(typeid(cl_double) == typeid(T))
        {
            for (int i = 0; i < hKeysOut.size(); i++)
            {
                ASSERT_EQ(hKeysOut[i], host_keys[i]);
                ASSERT_NEAR(hValsOut[i], host_values[i], 1e-14);
            }

        }




        status = clEnqueueUnmapMemObject(CLSE::queue, gValsOut.values, host_values,
                                         0, nullptr, nullptr);
        ASSERT_EQ(CL_SUCCESS, status);

        status = clEnqueueUnmapMemObject(CLSE::queue, gKeysOut.values, host_keys,
                                         0, nullptr, nullptr);
        ASSERT_EQ(CL_SUCCESS, status);


        clReleaseMemObject( gKeysOut.values );
        clReleaseMemObject( gValsOut.values );
    }

private:

    void randomize_keys()
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, BLAS1E::size/2);

        for (int i=0; i < BLAS1E::size; i++)
        {
            hKeys[i] = dis(gen);
            hValues[i] = 1;
        }

        std::sort(hKeys.begin(), hKeys.end());
    }
};

typedef ::testing::Types<cl_float, cl_double> TYPES;

TYPED_TEST_CASE(ReduceByKey, TYPES);

TYPED_TEST(ReduceByKey, reduce_by_key)
{
    this->test_reduce_by_key();
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

