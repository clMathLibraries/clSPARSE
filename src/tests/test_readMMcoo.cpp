/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

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

static cl_bool explicit_zeroes = true;

/** Just a simple test checking if the io functions for matrices are ok */

namespace po = boost::program_options;

std::string path;

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

    // Read sparse data from file and construct a COO matrix from it
    int nnz, row, col;
    clsparseStatus fileError = clsparseHeaderfromFile( &nnz, &row, &col, path.c_str( ) );
    if( fileError != clsparseSuccess )
        throw std::runtime_error( "Could not read matrix market header from disk" );

    // Now initialise a CSR matrix from the COO matrix
    clsparseCooMatrix cooMatx;
    clsparseInitCooMatrix( &cooMatx );
    cooMatx.num_nonzeros = nnz;
    cooMatx.num_rows = row;
    cooMatx.num_cols = col;

    cl_int status;
    cooMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.num_nonzeros * sizeof( cl_float ), NULL, &status );
    cooMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );
    cooMatx.rowIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );
    clsparseCooMatrixfromFile( &cooMatx, path.c_str( ), CLSE::control );

    clsparseCsrMatrix csrMatx;
    clsparseInitCsrMatrix( &csrMatx );

    csrMatx.values = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.num_nonzeros * sizeof( cl_float ), NULL, &status );
    csrMatx.colIndices = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                       cooMatx.num_nonzeros * sizeof( cl_int ), NULL, &status );
    csrMatx.rowOffsets = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                           ( cooMatx.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );
    //clsparseScoo2csr_host( &csrMatx, &cooMatx, CLSE::control );
    clsparseScoo2csr( &cooMatx, &csrMatx, CLSE::control );

    clsparseCsrMetaSize( &csrMatx, CLSE::control );

    csrMatx.rowBlocks = ::clCreateBuffer( CLSE::context, CL_MEM_READ_ONLY,
                                          csrMatx.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
    clsparseCsrMetaCompute( &csrMatx, CLSE::control );

    clsparseScalar alpha;
    clsparseScalar beta;
    cldenseVector x;
    cldenseVector y;
    clsparseInitScalar( &alpha );
    clsparseInitScalar( &beta );
    clsparseInitVector( &x );
    clsparseInitVector( &y );

    std::vector< cl_float > xHost( csrMatx.num_cols );
    std::vector< cl_float > yHost( csrMatx.num_rows );
    std::fill( xHost.begin( ), xHost.end( ), 1.0f );
    std::fill( yHost.begin( ), yHost.end( ), 2.0f );
    cl_float aHost = 1.0f;
    cl_float bHost = 1.0f;

    x.values = ::clCreateBuffer( CLSE::context,
                                CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                xHost.size( ) * sizeof( cl_float ), xHost.data( ), &status );
    x.num_values = yHost.size( );

    y.values = ::clCreateBuffer( CLSE::context,
                               CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                               yHost.size( ) * sizeof( cl_float ), yHost.data( ), &status );
    y.num_values = yHost.size( );

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
        "Path to matrix in mtx format." )
        ("no_zeroes,z", po::bool_switch()->default_value(false),
         "Disable reading explicit zeroes from the input matrix market file.");

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

    if (vm["no_zeroes"].as<bool>())
        explicit_zeroes = false;

    double alpha = 1.0;
    double beta = 1.0;

    ::testing::InitGoogleTest( &argc, argv );

    //order does matter!
    ::testing::AddGlobalTestEnvironment( new CLSE( ) );
    clsparseEnableExplicitZeroes(CLSE::control, explicit_zeroes);
    ::testing::AddGlobalTestEnvironment( new CSRE( path, alpha, beta,
        CLSE::queue, CLSE::context ) );

    return RUN_ALL_TESTS( );
}
