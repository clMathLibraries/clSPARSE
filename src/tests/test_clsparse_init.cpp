#include <clSPARSE.h>
#include <gtest/gtest.h>

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
    cl_uint major = 3, minor = 3, patch = 3;

    clsparseGetVersion (&major, &minor, &patch);

    EXPECT_EQ (0, major);
    EXPECT_EQ (1, minor);
    EXPECT_EQ (0, patch);
}

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
