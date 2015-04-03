#include <gtest/gtest.h>
#include <boost/program_options.hpp>

#include <clSPARSE.h>

#include "resources/clsparse_environment.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

namespace po = boost::program_options;

TEST (simple_kernel, run)
{
    using CLSE = ClSparseEnvironment;

//    clsparseControl control;
//    clsparseCreateControl(&control, CLSE::queue);

    size_t N = 1e7;

    cl_int status;
    cl_mem buff;

    //fix for CL 1.1 support (Nvidia devices)
    {
        cl_float value = 1.0;
        std::vector<cl_float> buff_value(N, value);

        buff = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                 N * sizeof(cl_float), buff_value.data(), &status);
        ASSERT_EQ(CL_SUCCESS, status);

        //not supported in CL 1.1
        //status = clEnqueueFillBuffer(CLSE::queue, buff, &value, sizeof(cl_float), 0,
        //                             N * sizeof(cl_float), 0, NULL, NULL);
    }
    ASSERT_EQ(CL_SUCCESS, status);

    cl_float halpha = 3.2;
    cl_mem alpha = clCreateBuffer(CLSE::context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float), &halpha, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    cl_event scale_event1;
    clsparseEnableAsync(CLSE::control, true);

    clsparseStatus clsp_status = clsparseScale(buff, alpha, N, CLSE::control);
    clsparseGetEvent(CLSE::control, &scale_event1);
    clWaitForEvents(1, &scale_event1);
    ASSERT_EQ(clsparseSuccess, clsp_status);
    ::clReleaseEvent( scale_event1 );


    std::vector<float> hbuff(N);

    clEnqueueReadBuffer(CLSE::queue, buff, true, 0,
                        N * sizeof(float), hbuff.data(), 0, NULL, NULL);


    for(int i = 0; i < N; i++)
    {
        EXPECT_FLOAT_EQ(halpha, hbuff[i]);
    }

}



int main(int argc, char* argv[])
{


    std::string platform;
    cl_uint device;

    po::options_description desc("Allowed options");

    desc.add_options()
            ("help,h", "Produce this message.")
            ("platform,l", po::value(&platform)->default_value("AMD"),
             "OpenCL platform: AMD or NVIDIA.")
            ("device,d", po::value(&device)->default_value(0),
             "Device id within platform.");

    po::variables_map vm;
    try{
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);
    } catch (po::error& error)
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what() << std::endl;
        std::cerr << desc << std::endl;
        exit(-1);
    }

    cl_platform_type p;
    //check platform
    if(vm.count("platform"))
    {
        if ("AMD" == platform)
        {
            p = AMD;
        }
        else if ("NVIDIA" == platform)
        {
            p = NVIDIA;
        }
        else
        {

            std::cout << "The platform option is missing or is ill defined!\n";
            std::cout << "Given [" << platform << "]" << std::endl;
            platform = "AMD";
            p = AMD;
            std::cout << "Setting [" << platform << "] as default" << std::endl;
        }

    }

    std::cout << "Using device = " << device
              << " within " << platform << " platform "<< std::endl;

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment( new ClSparseEnvironment(p, device));

    return RUN_ALL_TESTS();
}
