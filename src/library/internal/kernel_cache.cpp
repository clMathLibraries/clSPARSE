#include "kernel_cache.hpp"

#include <iostream>
#include <iterator>
#include "source_provider.hpp"

KernelCache KernelCache::singleton;

KernelCache::KernelCache()
{
    //we can add sth here which can be shared among all kernels;
}

cl::Kernel KernelCache::get(cl::CommandQueue& queue,
                         const std::string& program_name,
                         const std::string& kernel_name,
                         const std::string& params)
{
    return getInstance().getKernel(queue, program_name, kernel_name, params);
}


cl::Kernel KernelCache::getKernel(cl::CommandQueue& queue,
                                        const std::string& program_name,
                                        const std::string& kernel_name,
                                        const std::string& params)
{
    //!! ASSUMPTION: Kernel name == program name;
#if (BUILD_CLVERSION >= 120)
    std::string _params = " -cl-kernel-arg-info -cl-std=CL1.2 ";
#else
    std::string _params = " -cl-std=CL1.1 ";
#endif
    _params.append(params);
    std::string key;
    key.append( "[" + program_name + "/"  + kernel_name + "]");
    key.append(_params);

    auto hash = rsHash(key);
#ifndef NDEBUG
    std::cout << "key: " << key << " hash = " << hash << std::endl;
#endif

    auto kernel_iterator = kernel_map.find(hash);
    if (kernel_iterator != kernel_map.end())
    {

#ifndef NDEBUG
        std::cout << "kernel found: " << hash <<std::endl;
#endif
        return kernel_iterator->second;
    }
    else //build program and compile the kernel;
    {

        std::cout << "kernel not found: " << hash <<std::endl;
#ifndef NDEBUG
#endif

        const cl::Program* program = NULL;
        program = getProgram(queue, program_name, _params);
        if (program == nullptr)
        {
            std::cout << "Problem with getting program ["
                      << program_name << "] " << std::endl;
            delete program;
            return cl::Kernel();
        }

        cl_int status;
        cl::Kernel kernel(*program, kernel_name.c_str(), &status);

        if (status != CL_SUCCESS)
        {
            std::cout << "Problem with creating kernel ["
                      << kernel_name << "]" << std::endl;
            delete program;
            return cl::Kernel();
        }

        kernel_map[hash] = kernel;
        delete program;
        return kernel;
    }
}

const cl::Program* KernelCache::getProgram(cl::CommandQueue& queue,
                                         const std::string& program_name,
                                         const std::string& params)
{

    cl_int status;
    cl::Context context = queue.getInfo<CL_QUEUE_CONTEXT>();

    const char* source = SourceProvider::GetSource(program_name);
    if (source == nullptr)
    {
        std::cout << "Source not found [" << program_name << "]" << std::endl;
        return nullptr;
    }
    size_t size = std::char_traits<char>::length(source);

    cl::Program::Sources sources;

    std::pair<const char*, size_t> pair =
            std::make_pair(source, size);
    sources.push_back(pair);

    std::vector<cl::Device> devices;
    try
    {
       auto d = queue.getInfo<CL_QUEUE_DEVICE>();
       devices.push_back(d);

    } catch (cl::Error &e)
    {
        std::cout << "Problem with getting device form queue: "
                  << e.what() << " err: " << e.err() << std::endl;

        return nullptr;
    }


    cl::Program* program;

    try {
        program = new cl::Program(context, sources);
        program->build(devices, params.c_str());

    } catch (cl::Error& e)
    {

        std::cout << "#######################################" << std::endl;
        std::cout << "sources: ";
        for (auto& s : sources)
        {
            std::cout << s.first << std::endl;
        }

        std::cout << std::endl;

        std::cout << "---------------------------------------" << std::endl;
        std::cout << "parameters: " << params << std::endl;
        std::cout << "---------------------------------------" << std::endl;
        std::cout << program->getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[0])
                << std::endl;
        std::cout << "#######################################" << std::endl;

        return nullptr;
    }

    return program;
}



KernelCache& KernelCache::getInstance()
{
    return singleton;
}


unsigned int KernelCache::rsHash(const std::string &key)
{
    unsigned int b    = 378551;
    unsigned int a    = 63689;
    unsigned int hash = 0;
    unsigned int i    = 0;

    auto len = key.size();

    for (i = 0; i < len; i++)
    {
        hash = hash * a + key[i];
        a = a * b;
    }

    return hash;

}
