#include "../library/src/internal/clsparse_validate.h"
#include "clsparse_environment.h"

using CLSE = ClSparseEnvironment;

cl_command_queue CLSE::queue = NULL;
cl_context CLSE::context = NULL;

size_t N = 10;


//if somebody may want to store matrix as image we are not allowing that
TEST (clsparseValidateMem, not_flat_buffer)
{

    cl_int status;

    cl_image_format image_format;
    image_format.image_channel_order = CL_RGBA;
    image_format.image_channel_data_type = CL_FLOAT;

    cl_image_desc image_desc;
        image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
        image_desc.image_width = N;
        image_desc.image_height = N;
        image_desc.image_array_size = 1;
        image_desc.image_row_pitch = 0;
        image_desc.image_slice_pitch = 0;
        image_desc.num_mip_levels = 0;
        image_desc.num_samples = 0;
        image_desc.buffer = NULL;

    cl_mem mem = clCreateImage(CLSE::context, CL_MEM_READ_WRITE,
                               &image_format, &image_desc,
                              NULL, &status);
    ASSERT_EQ(CL_SUCCESS, status);

    status = validateMemObject(mem, N*N*sizeof(float));

    ASSERT_EQ(clsparseInvalidMemObj, status);

    clReleaseMemObject(mem);
}

TEST(clsparseValidateMem, flat_buffer)
{
    cl_int status;
    cl_mem mem = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                N * sizeof(cl_float), NULL, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    status = validateMemObject(mem, (N+1)*sizeof(cl_float));

    ASSERT_EQ(clsparseInvalidSize, status);

    status = validateMemObject(mem, (N-1)*sizeof(cl_float));

    ASSERT_EQ(clsparseInvalidSize, status);

    status = validateMemObject(mem, N*sizeof(cl_float));

    ASSERT_EQ(clsparseSuccess, status);

    clReleaseMemObject(mem);
}

TEST(clsparseValidateMemSize, zero_count)
{
    cl_int status;
    cl_mem mem = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                N * sizeof(cl_float), NULL, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    size_t count = 0;
    status = validateMemObjectSize(sizeof(cl_float), count, mem, 0);

    ASSERT_EQ(clsparseInvalidSize, status);

}

TEST(clsparseValidateMemSize, offset_overflow)
{
    cl_int status;
    cl_mem mem = clCreateBuffer(CLSE::context, CL_MEM_READ_WRITE,
                                N * sizeof(cl_float), NULL, &status);

    ASSERT_EQ(CL_SUCCESS, status);

    size_t count = 10;
    status = validateMemObjectSize(sizeof(cl_float), count, mem, 1);

    ASSERT_EQ(clsparseInsufficientMemory, status);

    count = 8;
    status = validateMemObjectSize(sizeof(cl_float), count, mem, 1);

    ASSERT_EQ(clsparseSuccess, status);

    status = validateMemObjectSize(sizeof(cl_float), count, mem, 2);

    ASSERT_EQ(clsparseSuccess, status);

    count = 8;
    status = validateMemObjectSize(sizeof(cl_float), count, mem, 3);

    ASSERT_EQ(clsparseInsufficientMemory, status);

}




int main(int argc, char* argv[])
{

    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment( new CLSE());

    return RUN_ALL_TESTS();
}
