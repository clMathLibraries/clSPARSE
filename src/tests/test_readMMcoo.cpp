#include <gtest/gtest.h>
#include <vector>
#include <string>
#include <boost/program_options.hpp>

#include "clSPARSE.hpp"
#include "resources/clsparse_environment.h"
#include "resources/csr_matrix_environment.h"
#include "resources/matrix_utils.h"

clsparseControl ClSparseEnvironment::control = NULL;
cl_command_queue ClSparseEnvironment::queue = NULL;
cl_context ClSparseEnvironment::context = NULL;

/** Just a simple test checking if the io functions for matrices are ok */

namespace po = boost::program_options;

std::string path;

class clMem12: public clAllocator
{
public:
    cl_mem_flags flags;
    void* hostBuffer;

    void* operator( )( size_t buffSize ) const
    {
        cl_mem buf;
        cl_int status;
        cl_context ctx = NULL;
            
        ::clGetCommandQueueInfo( queue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL );
        buf = ::clCreateBuffer( ctx, flags, buffSize, hostBuffer, &status );
        return buf;
    }
};

//class clMem20: public clAllocator
//{
//public:
//    cl_svm_mem_flags flags;
//
//    void* operator( )( size_t buffSize ) const
//    {
//        cl_context ctx = NULL;
//        
//        ::clGetCommandQueueInfo( queue, CL_QUEUE_CONTEXT, sizeof( cl_context ), &ctx, NULL );
//        return ::clSVMAlloc( ctx, flags, buffSize, 0 );
//    }
//};

void generateReference( const std::vector<float>& x,
                        const float alpha,
                        std::vector<float>& y,
                        const float beta )
{
    using CSRE = CSREnvironment;

    csrmv( CSRE::n_rows, CSRE::n_cols, CSRE::n_vals,
           CSRE::row_offsets, CSRE::col_indices, CSRE::f_values,
           x, alpha, y, beta );
}

TEST( MM_file, load )
{
    using CLSE = ClSparseEnvironment;

    clsparseCooMatrix cooMatx;
    clsparseInitCooMatrix( &cooMatx );

    clsparseCooHeaderfromFile( &cooMatx, path.c_str( ) );
    cl_int status;
    cooMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.nnz * sizeof( cl_float ), NULL, &status );
    cooMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
    cooMatx.rowIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.nnz * sizeof( cl_int ), NULL, &status );
    clsparseCooMatrixfromFile( &cooMatx, path.c_str( ), CLSE::control );

//#if( BUILD_CLVERSION < 200 )
//    clMem12 clAlloc;
//#else
//    clMem20 clAlloc;
//#endif
//
//    clAlloc.queue = CLSE::queue;
//    clAlloc.flags = CL_MEM_READ_ONLY;
//    clAlloc.hostBuffer = nullptr;
//    clsparseCooMatrixfromFile( &cooMatx, path.c_str( ), clAlloc );

    clsparseCsrMatrix csrMatx;
    clsparseInitCsrMatrix( &csrMatx );

    csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.nnz * sizeof( cl_float ), NULL, &status );
    csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.nnz * sizeof( cl_int ), NULL, &status );
    csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           ( cooMatx.m + 1 ) * sizeof( cl_int ), NULL, &status );
    clsparseScoo2csr( &csrMatx, &cooMatx, CLSE::control );

    clsparseCsrMetaSize( &csrMatx, CLSE::control );

    csrMatx.rowBlocks = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                          csrMatx.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
    clsparseCsrComputeMeta( &csrMatx, CLSE::control );

    clsparseScalar alpha;
    clsparseScalar beta;
    clsparseVector x;
    clsparseVector y;
    clsparseInitScalar( &alpha );
    clsparseInitScalar( &beta );
    clsparseInitVector( &x );
    clsparseInitVector( &y );

    std::vector< cl_float > xHost( csrMatx.n );
    std::vector< cl_float > yHost( csrMatx.m );
    std::fill( xHost.begin( ), xHost.end( ), 1.0f );
    std::fill( yHost.begin( ), yHost.end( ), 2.0f );
    cl_float aHost = 1.0f;
    cl_float bHost = 1.0f;

    x.values = ::clCreateBuffer( CLSE::context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                xHost.size( ) * sizeof( cl_float ), xHost.data( ), &status );
    x.n = yHost.size( );

    y.values = ::clCreateBuffer( CLSE::context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               yHost.size( ) * sizeof( cl_float ), yHost.data( ), &status );
    y.n = yHost.size( );

    alpha.value = ::clCreateBuffer( CLSE::context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof( cl_float ), &aHost, &status );

    beta.value = ::clCreateBuffer( CLSE::context,
                                  CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                  sizeof( cl_float ), &bHost, &status );

    clsparseScsrmv( &alpha, &csrMatx, &x, &beta, &y, CLSE::control );

    generateReference( xHost, aHost, yHost, bHost );

    std::vector<float> deviceResult( yHost.size( ) );
    ::clEnqueueReadBuffer( ClSparseEnvironment::queue,
                           y.values, CL_TRUE, 0,
                         deviceResult.size( )*sizeof( float ),
                         deviceResult.data( ), 0, NULL, NULL );

    for( int i = 0; i < yHost.size( ); ++i )
        ASSERT_NEAR( yHost[ i ], deviceResult[ i ], 5e-4 );

}


int main( int argc, char* argv[ ] )
{
    using CLSE = ClSparseEnvironment;
    using CSRE = CSREnvironment;

    po::options_description desc( "Allowed options" );

    desc.add_options( )
        ( "help,h", "Produce this message." )
        ( "path,p", po::value( &path )->required( ),
        "Path to matrix in mtx format." );

    po::variables_map vm;
    try
    {
        po::store( po::parse_command_line( argc, argv, desc ), vm );
        po::notify( vm );
    } catch( po::error& error )
    {
        std::cerr << "Parsing command line options..." << std::endl;
        std::cerr << "Error: " << error.what( ) << std::endl;
        std::cerr << desc << std::endl;
        return false;
    }

    double alpha = 1.0;
    double beta = 1.0;

    ::testing::InitGoogleTest( &argc, argv );

    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE( ) );
    ::testing::AddGlobalTestEnvironment( new CSRE( path, alpha, beta,
        CLSE::queue, CLSE::context ) );

    return RUN_ALL_TESTS( );
}
