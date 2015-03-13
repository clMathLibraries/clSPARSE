#include <gtest/gtest.h>

#include <clSPARSE.h>

#include "resources/clsparse_environment.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

TEST (simple_kernel, run)
{
    using CLSE = ClSparseEnvironment;

//    clsparseControl control;
//    clsparseCreateControl(&control, CLSE::queue);

    size_t N = 1024;

    cl_int status;
    cl_mem buff = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                 N * sizeof(cl_float), NULL, &status);
    ASSERT_EQ(CL_SUCCESS, status);

    cl_float value = 1.0;
    status = clEnqueueFillBuffer(CLSE::queue, buff, &value, sizeof(cl_float), 0,
                                 N * sizeof(cl_float), 0, NULL, NULL);

    ASSERT_EQ(CL_SUCCESS, status);

    cl_float halpha = 3.2;
    cl_mem alpha = clCreateBuffer(CLSE::context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof(cl_float), &halpha, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    cl_event scale_event;
    clsparseEventsToSync(CLSE::control, 0, NULL, &scale_event);
//    clsparseStatus clsp_status = clsparseScale(buff, alpha, N, CLSE::queue,
//                                               0, NULL, &scale_event);
    clsparseStatus clsp_status = clsparseScale(buff, alpha, N, CLSE::control);
    status = clsparseSynchronize(CLSE::control);
    //status = clWaitForEvents(1, &scale_event);

    ASSERT_EQ(CL_SUCCESS, status);
    ASSERT_EQ(clsparseSuccess, clsp_status);


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

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment( new ClSparseEnvironment());

    return RUN_ALL_TESTS();
}
