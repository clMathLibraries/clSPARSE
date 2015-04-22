#include <clSPARSE.h>
#include <gtest/gtest.h>
#include <iostream>

//this implementation is placed in #include "../library/internal/ocl_traits.hpp
template <typename T>
struct is_clmem
{
    typedef T value_type;

    static bool const value =
            (std::is_pointer<T>::value &&
             ! std::is_fundamental<typename std::remove_pointer<T>::type>::value);


};


TEST (CL_TRAITS, cl_mem_type)
{
    bool is_open_cl_type = is_clmem<cl_mem>::value;
    ASSERT_EQ(true, is_open_cl_type);
}

TEST (CL_TRAITS, non_cl_mem_type)
{
    bool is_open_cl_type = is_clmem<void*>::value;
    ASSERT_EQ(false, is_open_cl_type);
}

int main(int argc, char* argv[])
{
    //std::cout << "cl_mem = " << is_clmem<cl_mem>::value << std::endl;
    //std::cout << "void* = " << is_clmem<void*>::value << std::endl;
     ::testing::InitGoogleTest(&argc, argv);
     return RUN_ALL_TESTS();


	return 0;

}
