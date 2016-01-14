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

double Blas1Environment::alpha = 0;
double Blas1Environment::beta = 0;
int Blas1Environment::size = 0;

namespace po = boost::program_options;
namespace uBLAS = boost::numeric::ublas;


template <typename T>
class Blas1 : public ::testing::Test
{

     using CLSE = ClSparseEnvironment;
     using BLAS1E = Blas1Environment;


public:

    void SetUp()
    {
        clsparseInitScalar(&gAlpha);
        clsparseInitScalar(&gBeta);

        clsparseInitVector(&gX);
        clsparseInitVector(&gY);

        hY = uBLAS::vector<T>(BLAS1E::size, 2);
        hX = uBLAS::vector<T>(BLAS1E::size, 4);
        //std::iota(hX.begin(), hX.end(), 1);


        cl_int status;
        gX.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hX.size() * sizeof(T), hX.data().begin(), &status);
        gX.num_values = hX.size();
        ASSERT_EQ(CL_SUCCESS, status);


        gY.values = clCreateBuffer(CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   hY.size() * sizeof(T), hY.data().begin(), &status);

        gY.num_values = hY.size();
        ASSERT_EQ(CL_SUCCESS, status);


        gAlpha.value = clCreateBuffer(CLSE::context,
                                      CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(T), &hAlpha, &status);

        ASSERT_EQ(CL_SUCCESS, status);


        gBeta.value = clCreateBuffer(CLSE::context,
                                     CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     sizeof(T), &hBeta, &status);

        ASSERT_EQ(CL_SUCCESS, status);

    }


    void TearDown()
    {
       clReleaseMemObject( gX.values );
       clReleaseMemObject( gY.values );
       clReleaseMemObject( gAlpha.value );
       clReleaseMemObject( gBeta.value );
    }

    boost::numeric::ublas::vector<T> hY;
    boost::numeric::ublas::vector<T> hX;
    T hAlpha = BLAS1E::alpha;
    T hBeta = BLAS1E::beta;

    cldenseVector gX;
    cldenseVector gY;
    clsparseScalar gAlpha;
    clsparseScalar gBeta;


    /*  Test functions  */

    void test_reduce()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSreduce(&gAlpha, &gX, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDreduce(&gAlpha, &gX, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        T ublas_result = uBLAS::blas_1::asum(hX);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gAlpha.value,
                                              CL_TRUE, CL_MAP_READ,
                                              0, sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to function passing an assertion here;
        if ( typeid(T) == typeid(cl_float) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-7);
        if ( typeid(T) == typeid(cl_double) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gAlpha.value,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);
    }

    void test_norm1()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSnrm1(&gAlpha, &gX, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDnrm1(&gAlpha, &gX, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        T ublas_result = uBLAS::blas_1::asum(hX);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gAlpha.value,
                                              CL_TRUE, CL_MAP_READ,
                                              0, sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to function passing an assertion here;
        if ( typeid(T) == typeid(cl_float) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-7);
        if ( typeid(T) == typeid(cl_double) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gAlpha.value,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

    void test_norm2()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSnrm2(&gAlpha, &gX, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDnrm2(&gAlpha, &gX, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        T ublas_result = uBLAS::blas_1::nrm2(hX);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gAlpha.value,
                                              CL_TRUE, CL_MAP_READ,
                                              0, sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to function passing an assertion here;
        if ( typeid(T) == typeid(cl_float) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-7);
        if ( typeid(T) == typeid(cl_double) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gAlpha.value,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);
    }

    void test_scale()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSscale(&gX, &gAlpha, &gX, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDscale(&gX, &gAlpha, &gX, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        hX = uBLAS::blas_1::scal(hX, hAlpha);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gX.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gX.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hX.size(); i++)
                ASSERT_NEAR (hX[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hX.size(); i++)
                ASSERT_NEAR (hX[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gX.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

    void test_dot()
    {
        clsparseStatus status;
        if ( typeid(T) == typeid(cl_float) )
        {
            status = cldenseSdot(&gAlpha, &gX, &gY, CLSE::control);
        }
        else if( typeid(T)  == typeid (cl_double) )
        {
            status = cldenseDdot(&gAlpha, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        T ublas_result = uBLAS::blas_1::dot(hX, hY);

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gAlpha.value,
                                              CL_TRUE, CL_MAP_READ,
                                              0, sizeof(T), 0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to function passing an assertion here;
        if ( typeid(T) == typeid(cl_float) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-7);
        if ( typeid(T) == typeid(cl_double) )
            ASSERT_NEAR (ublas_result, *host_result, 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gAlpha.value,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);


    }

    void test_axpy()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSaxpy(&gY, &gAlpha, &gX, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDaxpy(&gY, &gAlpha, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        hY = uBLAS::blas_1::axpy(hY, hAlpha, hX);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);
    }


    void test_axpby()
    {
        clsparseStatus status;

        //  GPU result;
        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSaxpby(&gY, &gAlpha, &gX, &gBeta, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDaxpby(&gY, &gAlpha, &gX, &gBeta, &gY, CLSE::control);
        }

        ASSERT_EQ (clsparseSuccess, status);

        hY = uBLAS::blas_1::axpy(uBLAS::blas_1::scal(hY, hBeta), hAlpha, hX);

        // Map device data to host
        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

    void test_add()
    {
        clsparseStatus status;

        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSadd(&gY, &gX, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDadd(&gY, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ(clsparseSuccess, status);

        hY = hX + hY;

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);
    }

    void test_sub()
    {
        clsparseStatus status;

        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSsub(&gY, &gX, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDsub(&gY, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ(clsparseSuccess, status);

        hY = hX - hY;

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);



    }

    void test_mul()
    {
        clsparseStatus status;

        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSmul(&gY, &gX, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDmul(&gY, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ(clsparseSuccess, status);

        hY = uBLAS::element_prod(hX, hY);

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

    void test_div()
    {
        clsparseStatus status;

        if (typeid(T) == typeid(cl_float))
        {
            status = cldenseSdiv(&gY, &gX, &gY, CLSE::control);
        }
        else if ( typeid(T) == typeid(cl_double) )
        {
            status = cldenseDdiv(&gY, &gX, &gY, CLSE::control);
        }

        ASSERT_EQ(clsparseSuccess, status);

        hY = uBLAS::element_div(hX, hY);

        cl_int cl_status;
        T* host_result = (T*) ::clEnqueueMapBuffer(CLSE::queue, gY.values,
                                              CL_TRUE, CL_MAP_READ,
                                              0, gY.num_values * sizeof(T),
                                              0, nullptr, nullptr, &cl_status);
        ASSERT_EQ (CL_SUCCESS, cl_status);

        // change it to some compare template functions
        if ( typeid(T) == typeid(cl_float) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-7);

        if ( typeid(T) == typeid(cl_double) )
            for(int i = 0; i < hY.size(); i++)
                ASSERT_NEAR (hY[i], host_result[i], 1e-14);

        cl_status = ::clEnqueueUnmapMemObject(CLSE::queue, gY.values,
                                              host_result, 0, nullptr, nullptr);
        ASSERT_EQ (CL_SUCCESS, cl_status);

    }

};


typedef ::testing::Types<cl_float, cl_double> TYPES;
TYPED_TEST_CASE(Blas1, TYPES);

TYPED_TEST(Blas1, reduce)
{
    this->test_reduce();
}

TYPED_TEST(Blas1, norm1)
{
    this->test_norm1();
}

TYPED_TEST(Blas1, norm2)
{
    this->test_norm2();
}


TYPED_TEST(Blas1, scale)
{
    this->test_scale();
}

TYPED_TEST(Blas1, dot)
{
    this->test_dot();
}


TYPED_TEST(Blas1, axpy)
{
    this->test_axpy();
}

TYPED_TEST(Blas1, axpby)
{
    this->test_axpby();
}

TYPED_TEST(Blas1, add)
{
    this->test_add();
}

TYPED_TEST(Blas1, sub)
{
    this->test_sub();
}

TYPED_TEST(Blas1, mul)
{
    this->test_mul();
}

TYPED_TEST(Blas1, div)
{
    this->test_div();
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
        if (vm.count("help"))
        {
            std::cout << desc << std::endl;
            return 0;
        }
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
