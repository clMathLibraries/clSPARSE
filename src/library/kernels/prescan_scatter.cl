R"(

#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif

__kernel
void prescan_scatter ( __global int *key,
                       __global int *value,
                       __global int *scan_input,
                       const int size)
{
    const int i = get_global_id(0);

    if (i >= size) return;

    scan_input[key[i]] = value[i];
}
)"

