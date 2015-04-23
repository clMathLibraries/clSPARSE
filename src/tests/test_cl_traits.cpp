#include <clSPARSE.h>
#include <gtest/gtest.h>
#include <iostream>

#include "../library/internal/ocl_type_traits.hpp"


TEST (CL_TRAITS, cl_mem_type)
{
    bool is_fundamental = is_pointer_fundamental<cl_mem>::value;
    ASSERT_EQ(false, is_fundamental);
}

TEST (CL_TRAITS, non_cl_mem_type)
{
    bool is_fundamental = is_pointer_fundamental<void*>::value;
    ASSERT_EQ(true, is_fundamental);
}

int main(int argc, char* argv[])
{
    //std::cout << "cl_mem = " << is_clmem<cl_mem>::value << std::endl;
    //std::cout << "void* = " << is_clmem<void*>::value << std::endl;
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();


	return 0;

}
