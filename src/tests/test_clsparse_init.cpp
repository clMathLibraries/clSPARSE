#include <clSPARSE.h>
#include <gtest/gtest.h>

#include "opencl_utils.h"


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
    EXPECT_EQ (0, minor);
    EXPECT_EQ (1, patch);
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

int main(int argc, char* argv[])
{

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

