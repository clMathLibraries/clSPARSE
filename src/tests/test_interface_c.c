#include <clSPARSE.h>
#include <clSPARSE-error.h>

#include <assert.h>

#if defined(__APPLE__) || defined(__MACOSX)
#include <malloc/malloc.h>
#else
#include <malloc.h>
#endif

int main( int argc, char* argv[ ] )
{
    // init cl environment
    cl_int status = CL_SUCCESS;
    cl_platform_id* platforms = NULL;
    cl_device_id* devices = NULL;
    cl_uint num_platforms = 0;
    cl_uint num_devices = 0;

    // OpenCL initialization code
    status = clGetPlatformIDs( 0, NULL, &num_platforms );
    platforms = (cl_platform_id*) malloc( num_platforms * sizeof( cl_platform_id ) );

    status = clGetPlatformIDs( num_platforms, platforms, NULL );
    CLSPARSE_V( status, "clGetPlatformIDs" );

    //get count of given device types
    status = clGetDeviceIDs( platforms[ 0 ], CL_DEVICE_TYPE_DEFAULT, 0, NULL, &num_devices );
    CLSPARSE_V( status, "clGetDeviceIDs num_devices" );

    devices = (cl_device_id*) malloc( num_devices * sizeof( cl_device_id ) );
    status = clGetDeviceIDs( platforms[ 0 ], CL_DEVICE_TYPE_DEFAULT, num_devices, devices, NULL );
    CLSPARSE_V( status, "clGetDeviceIDs CL_DEVICE_TYPE_DEFAULT" );

    cl_context context = clCreateContext( NULL, 1, devices, NULL, NULL, NULL );
    cl_command_queue queue = clCreateCommandQueue( context, devices[ 0 ], 0, NULL );

    // Library init code starts here
    clsparseSetup( );

    clsparseControl control = clsparseCreateControl( queue, NULL );

    clsparseCooMatrix myCooMatx;
    clsparseInitCooMatrix( &myCooMatx );

    // Library termination
    clsparseReleaseControl( control );
    clsparseTeardown( );

    // OpenCL termination
    clReleaseCommandQueue( queue );
    clReleaseContext( context );

    free( devices );
    free( platforms );

    return 0;
}
