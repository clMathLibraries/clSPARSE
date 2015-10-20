/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
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

        clsparseInitCsrMatrix( &csrDMatrix );
        csrDMatrix.num_nonzeros = n_vals;
        csrDMatrix.num_rows = n_rows;
        csrDMatrix.num_cols = n_cols;

         //  Load single precision data from file; this API loads straight into GPU memory
        cl_int status;
        csrDMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrDMatrix.num_nonzeros * sizeof( cl_double ), NULL, &status );

        csrDMatrix.colIndices = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  csrDMatrix.num_nonzeros * sizeof( cl_int ), NULL, &status );

        csrDMatrix.rowOffsets = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                                  ( csrDMatrix.num_rows + 1 ) * sizeof( cl_int ), NULL, &status );

        clsparseStatus fileError = clsparseDCsrMatrixfromFile( &csrDMatrix, file_name.c_str( ), CLSE::control );
        if( fileError != clsparseSuccess )
            throw std::runtime_error( "Could not read matrix market data from disk: " + file_name );

        clsparseCsrMetaSize( &csrDMatrix, CLSE::control );
        csrDMatrix.rowBlocks = ::clCreateBuffer( context, CL_MEM_READ_WRITE,
                                                 csrDMatrix.rowBlockSize * sizeof( cl_ulong ), NULL, &status );
        clsparseCsrMetaCompute( &csrDMatrix, CLSE::control );


        //reassign the new matrix dimmesnions calculated clsparseCCsrMatrixFromFile to global variables
        n_vals = csrDMatrix.num_nonzeros;
        n_cols = csrDMatrix.num_cols;
        n_rows = csrDMatrix.num_rows;

        //  Download sparse matrix data to host
        //  First, create space on host to hold the data
        ublasDCsr = dMatrixType(n_rows, n_cols, n_vals);

        // This is nasty. Without that call ublasSCsr is not working correctly.
        ublasDCsr.complete_index1_data();

        // copy host matrix arrays to device;
        cl_int copy_status;

        copy_status = clEnqueueReadBuffer( queue, csrDMatrix.values, CL_TRUE, 0,
                                           csrDMatrix.num_nonzeros * sizeof( cl_double ),
                                           ublasDCsr.value_data().begin( ),
                                           0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrDMatrix.rowOffsets, CL_TRUE, 0,
                                            ( csrDMatrix.num_rows + 1 ) * sizeof( cl_int ),
                                            ublasDCsr.index1_data().begin(),
                                            0, NULL, NULL );

        copy_status = clEnqueueReadBuffer( queue, csrDMatrix.colIndices, CL_TRUE, 0,
                                            csrDMatrix.num_nonzeros * sizeof( cl_int ),
                                            ublasDCsr.index2_data().begin(),
                                            0, NULL, NULL );

        // Create matrix in float precision on host;
        ublasSCsr = sMatrixType(n_rows, n_cols, n_vals);
        ublasSCsr.complete_index1_data();

        // Create matrix in single precision on device;
        // Init single precision matrix;
        clsparseInitCsrMatrix( &csrSMatrix );

        csrSMatrix.num_nonzeros = csrDMatrix.num_nonzeros;
        csrSMatrix.num_cols = csrDMatrix.num_cols;
        csrSMatrix.num_rows = csrDMatrix.num_rows;
        csrSMatrix.rowBlockSize = csrDMatrix.rowBlockSize;

        // Don't use adaptive kernel in double precision yet.
        csrSMatrix.rowBlocks = csrDMatrix.rowBlocks;
        ::clRetainMemObject( csrSMatrix.rowBlocks );

        csrSMatrix.colIndices = csrDMatrix.colIndices;
        ::clRetainMemObject( csrSMatrix.colIndices );

        csrSMatrix.rowOffsets = csrDMatrix.rowOffsets;
        ::clRetainMemObject( csrSMatrix.rowOffsets );

        csrSMatrix.values = ::clCreateBuffer( context, CL_MEM_READ_ONLY,
                                              csrSMatrix.num_nonzeros * sizeof( cl_float ), NULL, &status );

        cl_int cl_status;
        cl_double* dvals = (cl_double*) ::clEnqueueMapBuffer(queue, csrDMatrix.values, CL_TRUE, CL_MAP_READ, 0, csrDMatrix.num_nonzeros * sizeof(cl_double), 0, nullptr, nullptr, &cl_status);

        // copy the double-precision values over into the single-precision array.
        for ( int i = 0; i < ublasDCsr.value_data().size(); i++)
            ublasSCsr.value_data()[i] = static_cast<double>(ublasDCsr.value_data()[i]);
        for ( int i = 0; i < ublasDCsr.index1_data().size(); i++)
            ublasSCsr.index1_data()[i] = static_cast<int>(ublasDCsr.index1_data()[i]);
        for ( int i = 0; i < ublasDCsr.index2_data().size(); i++)
            ublasSCsr.index2_data()[i] = static_cast<int>(ublasDCsr.index2_data()[i]);

        // copy the values in single precision on host to single precision matrix container on the device
        copy_status = clEnqueueWriteBuffer( queue, csrSMatrix.values, CL_TRUE, 0,
                                            csrSMatrix.num_nonzeros * sizeof( cl_float ),
                                            ublasSCsr.value_data().begin( ),
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
