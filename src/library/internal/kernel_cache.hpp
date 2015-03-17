#ifndef _KERNEL_CAHCE_HPP_
#define _KERNEL_CAHCE_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
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

    typedef std::map<unsigned int, cl::Kernel> KernelMap;

    static KernelCache& getInstance();

    static cl::Kernel get(cl::CommandQueue& queue,
                         const std::string& name,
                         const std::string& params = "");

    const cl::Program* getProgram(cl::CommandQueue& queue,
                              const std::string& name,
                              const std::string& params = "");

    cl::Kernel getKernel(cl::CommandQueue &queue,
                              const std::string& name,
                              const std::string& params = "");


private:


    unsigned int rsHash(const std::string& key);

    KernelMap kernel_map;

    KernelCache();

    static KernelCache singleton;
};

#endif //_KERNEL_CACHE_HPP_
