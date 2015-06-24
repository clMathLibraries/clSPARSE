#ifndef _CSR_MATRIX_ENVIRONMENT_H_
#define _CSR_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "clsparse_environment.h"

using CLSE = ClSparseEnvironment;

/**
 * @brief The CSREnvironment class will keep the input parameters for tests
 * They are list of paths to matrices in csr format in mtx files.
 */
class CSREnvironment: public ::testing::Environment
{
public:
    explicit CSREnvironment( const std::string& path,
                             double alpha, double beta,
                             cl_command_queue queue,
                             cl_context context ):
                             file_name( path ),
                             queue( queue ),
                             context( context )
    {
        clsparseStatus read_status = clsparseHeaderfromFile( &n_vals, &n_rows, &n_cols, file_name.c_str( ) );
        if( read_status )
        {
            exit( -3 );
        }

        clsparseInitCsrMatrix( &csrSMatrix );
        csrSMatrix.nnz = n_vals;
        csrSMatrix.m = n_rows;
        csrSMatrix.n = n_cols;
        clsparseCsrMetaSize( &csrSMatrix, CLSE::control );

        clsparseInitCsrMatrix( &csrDMatrix );
        csrDMatrix.nnz = n_vals;
        csrDMatrix.m = n_rows;
        csrDMatrix.n = n_cols;
        //clsparseCsrMetaSize( &csrDMatrix, CLSE::control );


        this->alpha = alpha;
        this->beta = beta;

    }


    void SetUp( )
    {
        //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;
        csrSMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrSMatrix.nnz * sizeof( cl_float ), NULL, &status );

        csrSMatrix.colIndices = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  csrSMatrix.nnz * sizeof( cl_int ), NULL, &status );

        csrSMatrix.rowOffsets = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  ( csrSMatrix.m + 1 ) * sizeof( cl_int ), NULL, &status );

        csrSMatrix.rowBlocks = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                 csrSMatrix.rowBlockSize * sizeof( cl_ulong ), NULL, &status );

        clsparseStatus fileError = clsparseSCsrMatrixfromFile( &csrSMatrix, file_name.c_str( ), CLSE::control );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

        //reassign the new matrix dimmesnions calculated clsparseCCsrMatrixFromFile to global variables
        n_vals = csrSMatrix.nnz;
        n_cols = csrSMatrix.n;
        n_rows = csrSMatrix.m;

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        f_values.resize( csrSMatrix.nnz );
        col_indices.resize( csrSMatrix.nnz );
        row_offsets.resize( csrSMatrix.m + 1 );

        // copy host matrix arrays to device;
        cl_int copy_status;
        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.values, CL_TRUE, 0,
                                           f_values.size( ) * sizeof( cl_float ),
                                           f_values.data( ),
                                           0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.rowOffsets, CL_TRUE, 0,
                                            row_offsets.size( ) * sizeof( int ),
                                            row_offsets.data( ),
                                            0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.colIndices, CL_TRUE, 0,
                                            col_indices.size( ) * sizeof( int ),
                                            col_indices.data( ),
                                            0, NULL, NULL );

        if( copy_status )
        {
            TearDown( );
            exit( -5 );
        }

        d_values = std::vector<double>( f_values.begin( ), f_values.end( ) );

        // Don't use adaptive kernel in double precision yet.
        //csrDMatrix.rowBlocks = csrSMatrix.rowBlocks;
        //::clRetainMemObject( csrDMatrix.rowBlocks );

        csrDMatrix.nnz = csrSMatrix.nnz;
        csrDMatrix.n = csrSMatrix.n;
        csrDMatrix.m = csrSMatrix.m;

        csrDMatrix.colIndices = csrSMatrix.colIndices;
        ::clRetainMemObject( csrDMatrix.colIndices );

        csrDMatrix.rowOffsets = csrSMatrix.rowOffsets;
        ::clRetainMemObject( csrDMatrix.rowOffsets );

        //  Intialize double precision data, all the indices can be reused.
        csrDMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrDMatrix.nnz * sizeof( cl_double ), NULL, &status );

        status = ::clEnqueueWriteBuffer( queue, csrDMatrix.values, CL_TRUE, 0,
                                              csrDMatrix.nnz * sizeof( cl_double ), d_values.data( ), 0, NULL, NULL );

        if( copy_status )
        {
            TearDown( );
            exit( -5 );
        }
    }

    //cleanup
    void TearDown( )
    {
        //release buffers;
        ::clReleaseMemObject( csrSMatrix.values );
        ::clReleaseMemObject( csrSMatrix.colIndices );
        ::clReleaseMemObject( csrSMatrix.rowOffsets );
        ::clReleaseMemObject( csrSMatrix.rowBlocks );
        ::clReleaseMemObject( csrDMatrix.values );
        ::clReleaseMemObject( csrDMatrix.colIndices );
        ::clReleaseMemObject( csrDMatrix.rowOffsets );
        ::clReleaseMemObject( csrDMatrix.rowBlocks );

        //bring csrSMatrix csrDMatrix to its initial state
        clsparseInitCsrMatrix( &csrSMatrix );
        clsparseInitCsrMatrix( &csrDMatrix );

    }


    static std::vector<int> row_offsets;
    static std::vector<int> col_indices;
    static std::vector<float> f_values;
    static std::vector<double> d_values;
    static int n_rows, n_cols, n_vals;

    //cl buffers for above matrix definition;

    static clsparseCsrMatrix csrSMatrix;
    static clsparseCsrMatrix csrDMatrix;

    static double alpha, beta;

private:
    cl_command_queue queue;
    cl_context context;
    std::string file_name;

};

#endif //_CSR_MATRIX_ENVIRONMENT_H_
