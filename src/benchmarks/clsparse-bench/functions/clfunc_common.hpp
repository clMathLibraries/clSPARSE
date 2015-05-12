/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/
#pragma once
#ifndef CLSPARSE_BENCHMARK_COMMON_HXX__
#define CLSPARSE_BENCHMARK_COMMON_HXX__

#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl_ext.h>
#endif

#include "clSPARSE.h"
#include "clsparse.error.hpp"

inline cl_ulong
queryMemAllocSize( cl_device_id device )
{
    cl_ulong rc = 0;
    cl_int err = clGetDeviceInfo( device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof( rc ), &rc, NULL );

    return rc;
}

class clsparseFunc
{
public:
    clsparseFunc( cl_device_type devType, cl_command_queue_properties cqProp ): cqProperties( cqProp )
    {
        cl_int err;

        // Setup OpenCL environment
        OPENCL_V_THROW( ::clGetPlatformIDs( 1, &platform, NULL ), "getting platform IDs" );
        OPENCL_V_THROW( ::clGetDeviceIDs( platform, devType, 1, &device, NULL ), "getting device IDs" );
        props[ 0 ] = CL_CONTEXT_PLATFORM;
        props[ 1 ] = (cl_context_properties)platform;
        props[ 2 ] = 0;

        ctx = ::clCreateContext( props, 1, &device, NULL, NULL, &err );
        OPENCL_V_THROW( err, "creating context" );

        queue = ::clCreateCommandQueue( ctx, device, cqProp, &err );
        OPENCL_V_THROW( err, "clCreateCommandQueue" );

        maxMemAllocSize = queryMemAllocSize( device );

        // Setup clsparse
        if( clsparseSetup( ) != clsparseSuccess )
        {
            std::cerr << "clsparseSetup() failed with %d\n";
            OPENCL_V_THROW( ::clReleaseCommandQueue( queue ), "releasing command queue" );
            OPENCL_V_THROW( ::clReleaseContext( ctx ), "releasing context" );
        }

        control = clsparseCreateControl( queue, NULL );

    }

    virtual ~clsparseFunc( )
    {
        if( clsparseReleaseControl( control ) != clsparseSuccess )
        {
            std::cout << "Problem with releasing control object" << std::endl;
        }

        clsparseTeardown( );
        OPENCL_V_THROW( ::clReleaseCommandQueue( queue ), "releasing command queue" );
        OPENCL_V_THROW( ::clReleaseContext( ctx ), "releasing context" );
    }

    void wait_and_check( )
    {
        cl_int wait_status = ::clWaitForEvents( 1, &event );

        if( wait_status != CL_SUCCESS )
        {
            if( wait_status == CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST )
            {
                cl_int err;
                clGetEventInfo( event, CL_EVENT_COMMAND_EXECUTION_STATUS,
                    sizeof( cl_int ), &err, NULL );
                std::cout << "blas function execution status error: " << err << std::endl;
            }
            else
            {
                std::cout << "blas function wait status error: " << wait_status << std::endl;
            }
        }
    }

    double time_in_ns( )
    {
        //clsparseTimer& timer = clsparseGetTimer( CLSPARSE_GPU );
        //return timer.getAverageTime( timer_id ) * 1e9;
        return 0.0;
    }

    virtual void call_func( ) = 0;
    virtual double gflops( ) = 0;
    virtual std::string gflops_formula( ) = 0;
    virtual double bandwidth( ) = 0;
    virtual std::string bandwidth_formula( ) = 0;
    virtual void setup_buffer( double alpha, double beta, const std::string& path ) = 0;
    virtual void initialize_cpu_buffer( ) = 0;
    virtual void initialize_gpu_buffer( ) = 0;
    virtual void reset_gpu_write_buffer( ) = 0;
    virtual void read_gpu_buffer( ) = 0;
    virtual void cleanup( ) = 0;

protected:
    cl_platform_id platform;
    cl_device_id device;
    cl_context_properties props[ 3 ];
    cl_context ctx;
    cl_command_queue queue;
    cl_command_queue_properties cqProperties;
    cl_event event;
    size_t maxMemAllocSize;

    clsparseControl control;
};

#endif // ifndef CLBLAS_BENCHMARK_COMMON_HXX__

