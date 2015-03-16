#ifndef _KERNEL_CAHCE_HPP_
#define _KERNEL_CAHCE_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.h>
#else
    #include <CL/cl.h>
#endif

#include <string>
#include <map>
/**
 * @brief The KernelCache class Build and cache kernels
 * singleton
 */
class KernelCache
{

public:

    typedef std::map<unsigned int, cl_kernel> KernelMap;

    static KernelCache& getInstance();

    static const cl_kernel get(cl_command_queue queue,
                               const std::string& name,
                               const std::string& params = "");

    const cl_program getProgram(cl_command_queue queue,
                              const std::string& name,
                              const std::string& params = "");

    const cl_kernel getKernel(cl_command_queue queue,
                              const std::string& name,
                              const std::string& params = "");


private:

    cl_int getBuildLog(cl_device_id device, cl_program program, const char* params);

    unsigned int rsHash(const std::string& key);

    KernelMap kernel_map;

    KernelCache();

    static KernelCache singleton;
};

#endif //_KERNEL_CACHE_HPP_
