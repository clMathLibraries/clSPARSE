#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "resources/clsparse_environment.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;

TEST (AXPY, float_simple)
{
    using CLSE = ClSparseEnvironment;

    cl_uint size = 1000;
    std::vector<cl_float> y(size, 1.0f);
    std::vector<cl_float> x(size, 1.0f);
    cl_float hAlpha = 2.0f;

    clsparseVector gX;
    clsparseVector gY;
    clsparseScalar gAlpha;
    clsparseInitVector(&gX);
    clsparseInitVector(&gY);
    clsparseInitScalar(&gAlpha);

    cl_int status;
    gAlpha.value = ::clCreateBuffer(CLSE::context,
                                  CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float), &hAlpha, &status);
    ASSERT_EQ(CL_SUCCESS, status);

    gX.values = ::clCreateBuffer(CLSE::context,
                                 CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 x.size() * sizeof(cl_float), x.data(), &status);
    gX.n = x.size();

    ASSERT_EQ(CL_SUCCESS, status);

    gY.values = ::clCreateBuffer(CLSE::context,
                                 CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR,
                                 y.size() * sizeof(cl_float), y.data(), &status);
    gY.n = y.size();

    ASSERT_EQ(CL_SUCCESS, status);

    status = clsparseSaxpy(&gY, &gAlpha, &gX, CLSE::control);

    ASSERT_EQ(clsparseSuccess, status);

    for (int i = 0; i < size; i++)
    {
        y[i] = hAlpha * x[i] + y[i];
    }

    std::vector<cl_float> result(size);

    clEnqueueReadBuffer(CLSE::queue,
                        gY.values, 1, 0,
                        size*sizeof(cl_float),
                        result.data(), 0, NULL, NULL);

    for (int i = 0; i < size; i++)
    {
        ASSERT_NEAR(y[i], result[i], 5e-8);
    }
}

int main (int argc, char* argv[])
{

    using CLSE = ClSparseEnvironment;


    std::string platform;
    cl_platform_type pID;
    cl_uint dID;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&dID)->default_value(0),
             "Device id within platform.");


    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    }
    catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }


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
   
    return RUN_ALL_TESTS();

}
