#ifndef _CSR_MATRIX_ENVIRONMENT_H_
#define _CSR_MATRIX_ENVIRONMENT_H_

#include <gtest/gtest.h>
#include <clSPARSE.h>

#include "clsparse_environment.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
using CLSE = ClSparseEnvironment;

namespace uBLAS = boost::numeric::ublas;

/**
 * @brief The CSREnvironment class will keep the input parameters for tests
 * They are list of paths to matrices in csr format in mtx files.
 */
class CSREnvironment: public ::testing::Environment
{
public:

    // We need this long declaration because index vector need to be cl_int.
    // Also it is more flexible for future use if we will start to play with
    // row_major / column_major or base indexing which is 0 for now.
    using sMatrixType = uBLAS::compressed_matrix<cl_float,  uBLAS::row_major, 0, uBLAS::unbounded_array<int> >;
    using dMatrixType = uBLAS::compressed_matrix<cl_double, uBLAS::row_major, 0, uBLAS::unbounded_array<int> >;

    explicit CSREnvironment( const std::string& path,
                             cl_double alpha, cl_double beta,
                             cl_command_queue queue,
                             cl_context context ):
                             queue( queue ),
                             context( context )
    {
        file_name = path;
        clsparseStatus read_status = clsparseHeaderfromFile( &n_vals, &n_rows, &n_cols, file_name.c_str( ) );
        if( read_status )
        {
            exit( -3 );
        }

        clsparseInitCsrMatrix( &csrSMatrix );
        csrSMatrix.num_nonzeros = n_vals;
        csrSMatrix.num_rows = n_rows;
        csrSMatrix.num_cols = n_cols;
        clsparseCsrMetaSize( &csrSMatrix, CLSE::control );

         //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;
        csrSMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrSMatrix.num_nonzeros * sizeof( cl_float ), NULL, &status );

        csrSMatrix.colIndices = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  csrSMatrix.num_nonzeros * sizeof( cl_int ), NULL, &status );

        csrSMatrix.rowOffsets = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  ( csrSMatrix.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );

        csrSMatrix.rowBlocks = ::clCreateBuffer( context, CL_MEM_READ_WRITE,
                                                 csrSMatrix.rowBlockSize * sizeof( cl_ulong ), NULL, &status );

        clsparseStatus fileError = clsparseSCsrMatrixfromFile( &csrSMatrix, file_name.c_str( ), CLSE::control );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk" );

        //reassign the new matrix dimmesnions calculated clsparseCCsrMatrixFromFile to global variables
        n_vals = csrSMatrix.num_nonzeros;
        n_cols = csrSMatrix.num_cols;
        n_rows = csrSMatrix.num_rows;

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        ublasSCsr = sMatrixType(n_rows, n_cols, n_vals);

        // This is nasty. Without that call ublasSCsr is not working correctly.
        ublasSCsr.complete_index1_data();

        // copy host matrix arrays to device;
        cl_int copy_status;

        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.values, CL_TRUE, 0,
                                           csrSMatrix.num_nonzeros * sizeof( cl_float ),
                                           ublasSCsr.value_data().begin( ),
                                           0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.rowOffsets, CL_TRUE, 0,
                                            ( csrSMatrix.num_rows + 1 ) * sizeof( cl_int ),
                                            ublasSCsr.index1_data().begin(),
                                            0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrSMatrix.colIndices, CL_TRUE, 0,
                                            csrSMatrix.num_nonzeros * sizeof( cl_int ),
                                            ublasSCsr.index2_data().begin(),
                                            0, NULL, NULL );

        // Create matrix in double precision on host;
        ublasDCsr = dMatrixType(n_rows, n_cols, n_vals);
        ublasDCsr.complete_index1_data();

        // Create matrix in double precision on device;
        // Init double precision matrix;
        clsparseInitCsrMatrix( &csrDMatrix );

        csrDMatrix.num_nonzeros = csrSMatrix.num_nonzeros;
        csrDMatrix.num_cols = csrSMatrix.num_cols;
        csrDMatrix.num_rows = csrSMatrix.num_rows;
        csrDMatrix.rowBlockSize = csrSMatrix.rowBlockSize;

        // Don't use adaptive kernel in double precision yet.
        csrDMatrix.rowBlocks = csrSMatrix.rowBlocks;
        ::clRetainMemObject( csrDMatrix.rowBlocks );

        csrDMatrix.colIndices = csrSMatrix.colIndices;
        ::clRetainMemObject( csrDMatrix.colIndices );

        csrDMatrix.rowOffsets = csrSMatrix.rowOffsets;
        ::clRetainMemObject( csrDMatrix.rowOffsets );

        csrDMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrDMatrix.num_nonzeros * sizeof( cl_double ), NULL, &status );

        // copy the single-precision values over into the double-precision array.
        for ( int i = 0; i < ublasSCsr.value_data().size(); i++)
            ublasDCsr.value_data()[i] = static_cast<double>(ublasSCsr.value_data()[i]);
        for ( int i = 0; i < ublasSCsr.index1_data().size(); i++)
            ublasDCsr.index1_data()[i] = static_cast<int>(ublasSCsr.index1_data()[i]);
        for ( int i = 0; i < ublasSCsr.index2_data().size(); i++)
            ublasDCsr.index2_data()[i] = static_cast<int>(ublasSCsr.index2_data()[i]);

        // copy the values in double precision to double precision matrix container
        copy_status = clEnqueueWriteBuffer( queue, csrDMatrix.values, CL_TRUE, 0,
                                            csrDMatrix.num_nonzeros * sizeof( cl_double ),
                                            ublasDCsr.value_data().begin( ),
                                            0, NULL, NULL);

        if( copy_status )
        {
            TearDown( );
            exit( -5 );
        }

        this->alpha = alpha;
        this->beta = beta;

    }


    void SetUp( )
    {

        // Prepare data to it's default state


    }

    //cleanup
    void TearDown( )
    {


    }

    std::string getFileName()
    {
        return file_name;
    }

    ~CSREnvironment()
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

    static sMatrixType ublasSCsr;
    static dMatrixType ublasDCsr;

    static cl_int n_rows, n_cols, n_vals;

    //cl buffers for above matrix definition;

    static clsparseCsrMatrix csrSMatrix;
    static clsparseCsrMatrix csrDMatrix;

    static cl_double alpha, beta;
    static std::string file_name;

private:
    cl_command_queue queue;
    cl_context context;

};

#endif //_CSR_MATRIX_ENVIRONMENT_H_
