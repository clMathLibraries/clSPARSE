#pragma once
#ifndef _CLSPARSE_SOURCES_H_
#define _CLSPARSE_SOURCES_H_

#include "clsparse_internal.hpp"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


// creates program map key by concatenating name + params
void createKey( const char* name, const char* params, char** key );

cl_int printBuildLog( cl_device_id device, cl_program program, const char* params );

//TODO: Make it clear and nice!
cl_program buildProgram( cl_command_queue queue,
                            const char* name, const char* params,
                            const char* key,
                            cl_int* status );

// Assume kernel name == program name
// gets the kernel from cache, if not in cache build and append
// exceptionaly this name of the function is written as get_kernel due
// to conflicting symbol from getKernel from clBLAS ! nasty!
cl_kernel get_kernel( cl_command_queue queue,
                        const char* program_name, const char* params,
                        const char* key, cl_int* status );


#ifdef __cplusplus
}
#endif


#endif //_CLSPARSE_SOURCES_H_
