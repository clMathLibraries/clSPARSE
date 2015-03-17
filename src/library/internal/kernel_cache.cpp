#include "kernel_cache.hpp"

#include <iostream>
#include <iterator>
#include "source_provider.hpp"

KernelCache KernelCache::singleton;

KernelCache::KernelCache()
{
    //we can add sth here which can be shared among all kernels;
}


cl_kernel KernelCache::get(cl_command_queue& queue,
                                 const std::string& name,
                                 const std::string& params)
{
    return getInstance().getKernel(queue, name, params);
}

cl_kernel KernelCache::getKernel(cl_command_queue& queue,
                                        const std::string& name,
                                        const std::string& params)
{
    //!! ASSUMPTION: Kernel name == program name;
    std::string _params = "-Werror -cl-kernel-arg-info -cl-std=CL1.2 ";
    _params.append(params);
    std::string key;
    key.append(name);
    key.append(_params);

    auto hash = rsHash(key);
#ifndef NDEBUG
    std::cout << "key: " << key << " hash = " << hash << std::endl;
#endif

    auto kernel_iterator = kernel_map.find(hash);
    if (kernel_iterator != kernel_map.end())
    {
#ifndef NDEBUG
        std::cout << "kernel found" << std::endl;
#endif
        return kernel_iterator->second;
    }
    else //build program and compile the kernel;
    {
        cl_program program = getProgram(queue, name, _params);
        if (program == nullptr)
        {
            return nullptr;
        }

        cl_int status;
        cl_kernel kernel = clCreateKernel(program, name.c_str(), &status);

        if (status != CL_SUCCESS)
        {
            std::cout << "Problem with creating kernel ["
                      << kernel << "]" << std::endl;
            clReleaseProgram(program);
            return nullptr;
        }

        kernel_map[hash] = kernel;

        clReleaseProgram(program);
        return kernel;
    }
}

const cl_program KernelCache::getProgram(cl_command_queue& queue,
                                         const std::string& name,
                                         const std::string& params)
{
    //TODO:: Add context and decvice to the instance;
    cl_int status;
    cl_context context;
    status = clGetCommandQueueInfo(queue, CL_QUEUE_CONTEXT,
                                   sizeof(context), &context, NULL);
    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with obtaining queue context" << std::endl;
        return nullptr;
    }

    const char* source = SourceProvider::GetSource(name);
    if (source == nullptr)
    {
        std::cout << "Source not found [" << name << "]" << std::endl;
        return nullptr;
    }

    size_t size = std::char_traits<char>::length(source);

    cl_program program =
            clCreateProgramWithSource(context, 1, &source, &size, &status);

    if (status != CL_SUCCESS)
    {
        std::cout << "Problem creating program with source";
        return nullptr;
    }

    cl_device_id device;
    status = clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
                                   sizeof(device), &device, NULL);

    if (status != CL_SUCCESS)
    {
        std::cout << "Problem with getting queue" << std::endl;
        return nullptr;
    }

    status = clBuildProgram(program, 1, &device,
                            params.c_str(), NULL, NULL);

    if (status != CL_SUCCESS)
    {
        cl_int log_status = getBuildLog(device, program, params.c_str());
        if (log_status != CL_SUCCESS)
        {
            std::cerr << "FATAL!: Problem with generating build log"
                      << "for program " << name << std::endl;
        }
        return nullptr;
    }

    return program;
}

cl_int KernelCache::getBuildLog(cl_device_id& device,
                                cl_program& program, const char *params)
{
    cl_int status;
    char* log;
    size_t log_size;
    char* source;
    size_t source_size;

    status = clGetProgramInfo( program, CL_PROGRAM_SOURCE, 0, NULL, &source_size );
    if( status != CL_SUCCESS )
    {
        return status;
    }

    source = (char*) malloc( source_size * sizeof( char ) );

    status = clGetProgramInfo( program, CL_PROGRAM_SOURCE, source_size, source, NULL );
    if( status != CL_SUCCESS )
    {
        return status;
    }


    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size );
    if( status != CL_SUCCESS )
    {
        return status;
    }


    log = (char*) malloc( log_size * sizeof( char ) );

    status = clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL );
    if( status != CL_SUCCESS )
    {
        return status;
    }

    printf( "############ Build Log ############\n" );
    printf( "Params: %s\n", params );
    printf( "Source: \n%s\n", source );
    printf( "ERROR: %s\n", log );
    free( log );
    free( source );
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
