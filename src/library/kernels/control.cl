R"(
//do not remove. This kernel is used to measure some parameters of the device
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

#ifndef WG_SIZE
#error WG_SIZE undefined!
#endif

__kernel
__attribute__((reqd_work_group_size(WG_SIZE,1,1)))
void control(void)
{
    return;
}
)"
