#include <clSPARSE.h>
#include <gtest/gtest.h>

#include "opencl_utils.h"
#include "../library/internal/ocl-type-traits.hpp"

TEST (clSparseTraits, cl_mem_type)
{
    bool is_fundamental = is_pointer_fundamental<cl_mem>::value;
    ASSERT_EQ(false, is_fundamental);
}

TEST (clSparseTraits, non_cl_mem_type)
{
    bool is_fundamental = is_pointer_fundamental<void*>::value;
    ASSERT_EQ(true, is_fundamental);
}


TEST (clSparseInit, setup)
{
    clsparseStatus status = clsparseSetup();

    EXPECT_EQ(clsparseSuccess, status);
}

TEST (clSparseInit, teardown)
{
    clsparseSetup();
    clsparseStatus status = clsparseTeardown();

    EXPECT_EQ (clsparseSuccess, status);
}

TEST (clSparseInit, version)
{
    cl_uint major = 3, minor = 3, patch = 3, tweak = 3;

    clsparseGetVersion (&major, &minor, &patch, &tweak );

    EXPECT_EQ (0, major);
    EXPECT_EQ (6, minor);
    EXPECT_EQ (0, patch);
    EXPECT_EQ( 0, tweak );
}

TEST (clsparseInit, control)
{

    // init cl environment
    cl_int status = CL_SUCCESS;
    cl_platform_id* platforms = NULL;
    cl_uint num_platforms = 0;

    status = getPlatforms(&platforms, &num_platforms);
    ASSERT_EQ(CL_SUCCESS, status);

    //printPlatforms(platforms, num_platforms);

    cl_device_id device = NULL;
    status = getDevice(platforms[0], &device, CL_DEVICE_TYPE_GPU);
    ASSERT_EQ(CL_SUCCESS, status);

    //printDeviceInfo(device);

    auto context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    auto queue = clCreateCommandQueue(context, device, 0, NULL);

    clsparseSetup( );

    auto control = clsparseCreateControl(queue, NULL);

    clsparseReleaseControl( control );
    clsparseTeardown( );
    ::clReleaseCommandQueue( queue );
    ::clReleaseContext( context );

    free( platforms );

}

TEST (clsparseInit, cpp_interface)
{
    // Init OpenCL environment;
    cl_int cl_status;

    // Get OpenCL platforms
    std::vector<cl::Platform> platforms;

    cl_status = cl::Platform::get(&platforms);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting OpenCL platforms"
                  << " [" << cl_status << "]" << std::endl;
        ASSERT_EQ(CL_SUCCESS, cl_status);
    }

    int platform_id = 0;
    for (const auto& p : platforms)
    {
        std::cout << "Platform ID " << platform_id++ << " : "
                  << p.getInfo<CL_PLATFORM_NAME>() << std::endl;

    }

    // Using first platform
    platform_id = 0;
    cl::Platform platform = platforms[platform_id];

    // Get device from platform
    std::vector<cl::Device> devices;
    cl_status = platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (cl_status != CL_SUCCESS)
    {
        std::cout << "Problem with getting devices from platform"
                  << " [" << platform_id << "] " << platform.getInfo<CL_PLATFORM_NAME>()
                  << " error: [" << cl_status << "]" << std::endl;
        ASSERT_EQ(CL_SUCCESS, cl_status);
    }

    std::cout << std::endl
              << "Getting devices from platform " << platform_id << std::endl;

    cl_int device_id = 0;
    for (const auto& device : devices)
    {
        std::cout << "Device ID " << device_id++ << " : "
                  << device.getInfo<CL_DEVICE_NAME>() << std::endl;

    }

    // Using first device;
    device_id = 0;
    cl::Device device = devices[device_id];

    // Create OpenCL context;
    cl::Context context (device);

    // Create OpenCL queue;
    cl::CommandQueue queue(context, device);

    clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cout << "Problem with executing clsparseSetup()" << std::endl;
        ASSERT_EQ(clsparseSuccess, status);
    }


    // Create clsparseControl object
    clsparseControl control = clsparseCreateControl(queue(), &status);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with creating clSPARSE control object"
                  <<" error [" << status << "]" << std::endl;
        ASSERT_EQ(clsparseSuccess, status);
    }

    //cleanup;
    status = clsparseReleaseControl(control);
    ASSERT_EQ(clsparseSuccess, status);


    status = clsparseTeardown();
    ASSERT_EQ(clsparseSuccess, status);

}

int main(int argc, char* argv[])
{

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
