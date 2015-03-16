#include "clsparse_sources.hpp"

// creates program map key by concatenating name + params
void createKey( const char* name, const char* params, char** key )
{
    size_t key_size = strlen( params ) + strlen( name ) + 1;

#ifndef NDEBUG
    printf( "Creating key for %s, %s\n", name, params );
#endif
    //char* key = malloc(key_size*sizeof(char)); // REMEMBER TO FREE IT!
    *key = (char*) calloc( key_size, sizeof( char ) );

    //printf("K0 = %s %lu\n", *key, strlen(*key));
    strcpy( *key, name );
    //printf("K1 = %s %lu\n", *key, strlen(*key));
    strcat( *key, params );
    //printf("K2 = %s %lu\n", *key, strlen(*key));

#ifndef NDEBUG
    const unsigned int hash = RSHash( *key, strlen( *key ) );
    printf( "HASH: %u\n", hash );
#endif

}

cl_int printBuildLog( cl_device_id device, cl_program program, const char* params )
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

//TODO: Make it clear and nice!
cl_program buildProgram( cl_command_queue queue,
                         const char* name, const char* params,
                         const char* key,
                         cl_int* status )
{
    cl_program program;

    cl_context context;
    *status = clGetCommandQueueInfo( queue, CL_QUEUE_CONTEXT,
                                     sizeof( context ), &context, NULL );
    if( *status != CL_SUCCESS )
    {
        //free(key);
        //return status;
        printf( "Problem with obtaining context.\n" );
        return NULL;
    }

    //TODO: make separate h file for keys and sources.
    //Don't want to include sources to all files!
    const char* source = (const char*) hdl_get_value_by_key( program_sources, name );
    if( source == NULL )
    {
        printf( "Problem with obtaining kenrel source.\n" );
        return NULL;
    }
    const size_t size = strlen( source );

    program = clCreateProgramWithSource( context, 1, &source, &size, status );

    if( *status != CL_SUCCESS )
    {
        //free(key);
        //return status;
        printf( "Problem creating program with source" );
        return NULL;
    }

    cl_device_id device;
    *status = clGetCommandQueueInfo( queue, CL_QUEUE_DEVICE,
                                     sizeof( device ), &device, NULL );

    if( *status != CL_SUCCESS )
    {
        //        free(key);
        //        return status;
        printf( "Problem with getting queue device\n" );
    }

    *status = clBuildProgram( program, 1, &device, params, NULL, NULL );

    if( *status != CL_SUCCESS )
    {
#ifndef NDEBUG
        cl_int status2 = printBuildLog( device, program, params );
        if( status2 != CL_SUCCESS )
        {
            printf( "Fatal!: Problem with generating build log for program!\n" );
        }
#endif

        printf( "Problem with building program\n" );
        return NULL;
    }

    return program;
}

// Assume kernel name == program name
// gets the kernel from cache, if not in cache build and append
// exceptionaly this name of the function is written as get_kernel due
// to conflicting symbol from getKernel from clBLAS ! nasty!
cl_kernel get_kernel( cl_command_queue queue,
                      const char* program_name, const char* params,
                      const char* key, cl_int* status )
{
    *status = CL_SUCCESS;
    //for sake of clarity
    const char* kernel_name = program_name;

    //get kernel from cache
    const hdl_element* e = hdl_get_element_by_key( kernel_cache, key );

    //if not present build and add to cache
    if( e == NULL )
    {

#ifndef NDEBUG
        printf( "Building: %s with\n%s\n", program_name, params );
#endif
        cl_program program = buildProgram( queue, program_name,
                                           params, key, status );
        if( *status != CL_SUCCESS )
        {
            return NULL;
        }

        cl_kernel kernel = clCreateKernel( program, kernel_name, status );
        if( *status != CL_SUCCESS )
        {
            clReleaseProgram( program );
            return NULL;
        }
        // here we are passing kernel pointer to our hdl_list.
        // kernel cahce will manage to remove the kernel obejcts from memory
        hdl_insert( kernel_cache, key, kernel );

        clReleaseProgram( program );
        return kernel;
    }

    else
    {
        return (cl_kernel) e->value;
    }

}


