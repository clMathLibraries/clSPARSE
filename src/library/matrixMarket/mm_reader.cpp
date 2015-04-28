/*
Portions of this file include code provided by The National Institute of
Standards and Technology (NIST).  The code includes
macro definitions from mmio.h and is subject to the following disclaimer.

Software Disclaimer

NIST-developed software is provided by NIST as a public service. You may use,
copy and distribute copies of the software in any medium, provided that you
keep intact this entire notice. You may improve, modify and create derivative
works of the software or any portion of the software, and you may copy and
distribute such modifications or works. Modified works should carry a notice
stating that you changed the software and should note the date and nature of
any such change. Please explicitly acknowledge the National Institute of
Standards and Technology as the source of the software.

NIST-developed software is expressly provided "AS IS" NIST MAKES NO WARRANTY
OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW,
INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST
NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE
UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES
NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR
THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY,
RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

You are solely responsible for determining the appropriateness of using and
distributing the software and you assume all risks associated with its use,
including but not limited to the risks and costs of program errors, compliance
with applicable laws, damage to or loss of data, programs or equipment, and
the unavailability or interruption of operation. This software is not intended
to be used in any situation where a failure could cause risk of injury or
damage to property. The software developed by NIST employees is not subject
to copyright protection within the United States.
*/

#include <string>
#include <cstring>
#include <fstream>
#include <cstdio>
#include <iostream>
#include <typeinfo>

#include "include/clSPARSE-private.hpp"
#include "include/external/mmio.h"
#include "clSPARSE.hpp"
#include "internal/clsparse_control.hpp"

// Class declarations
template<typename FloatType>
struct Coordinate
{
    int x;
    int y;
    FloatType val;
};

template <typename FloatType>
class MatrixMarketReader
{
    char Typecode[ 4 ];
    int nNZ;
    int nRows;
    int nCols;
    int isSymmetric;
    int isDoubleMem;
    Coordinate<FloatType> *unsym_coords;

public:
    MatrixMarketReader( ): nNZ( 0 ), nRows( 0 ), nCols( 0 ), isSymmetric( 0 ), isDoubleMem( 0 )
    {
        for( auto c : Typecode )
            c = '\0';

        unsym_coords = NULL;
    }

    bool MMReadHeader( FILE* infile );
    bool MMReadHeader( const std::string& filename );
    bool MMReadFormat( const std::string& _filename );
    int MMReadBanner( FILE* infile );
    int MMReadMtxCrdSize( FILE* infile );
    void MMGenerateCOOFromFile( FILE* infile );

    int GetNumRows( )
    {
        return nRows;
    }

    int GetNumCols( )
    {
        return nCols;
    }

    int GetNumNonZeroes( )
    {
        return nNZ;
    }

    int GetSymmetric( )
    {
        return isSymmetric;
    }

    char &GetTypecode( )
    {
        return Typecode;
    }

    Coordinate<FloatType> *GetUnsymCoordinates( )
    {
        return unsym_coords;
    }

    ~MatrixMarketReader( )
    {
        delete[ ] unsym_coords;
    }
};

// Class definition

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadHeader( FILE* mm_file )
{
    int status = MMReadBanner( mm_file );
    if( status != 0 )
    {
        printf( "Error Reading Banner in Matrix-Market File !\n" );
        return 1;
    }

    if( !mm_is_coordinate( Typecode ) )
    {
        printf( " only handling coordinate format\n" ); return( 1 );
    }

    if( mm_is_complex( Typecode ) )
    {
        printf( "Error: cannot handle complex format\n" );
        return ( 1 );
    }

    if( mm_is_symmetric( Typecode ) )
        isSymmetric = 1;

    status = MMReadMtxCrdSize( mm_file );
    if( status != 0 )
    {
        printf( "Error reading Matrix Market crd_size %d\n", status );
        return( 1 );
    }

    return 0;
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadHeader( const std::string &filename )
{
    FILE *mm_file = ::fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        printf( "Cannot Open Matrix-Market File !\n" );
        return 1;
    }

    MMReadHeader( mm_file );

    // If symmetric MM stored file, double the reported size
    if( mm_is_symmetric( Typecode ) )
        nNZ <<= 1;

    ::fclose( mm_file );

    std::clog << "Matrix: " << filename << " [nRow: " << GetNumRows( ) << "] [nCol: " << GetNumCols( ) << "] [nNZ: " << GetNumNonZeroes( ) << "]" << std::endl;

    return 0;
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadFormat( const std::string &filename )
{
    FILE *mm_file = ::fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        printf( "Cannot Open Matrix-Market File !\n" );
        return 1;
    }

    MMReadHeader( mm_file );

    if( mm_is_symmetric( Typecode ) )
        unsym_coords = new Coordinate<FloatType>[ 2 * nNZ ];
    else
        unsym_coords = new Coordinate<FloatType>[ nNZ ];

    MMGenerateCOOFromFile( mm_file );
    ::fclose( mm_file );

    return 0;
}

template<typename FloatType>
void FillCoordData( char Typecode[ ],
                    Coordinate<FloatType> *unsym_coords,
                    int &unsym_actual_nnz,
                    int ir,
                    int ic,
                    FloatType val )
{
    if( mm_is_symmetric( Typecode ) )
    {
        unsym_coords[ unsym_actual_nnz ].x = ir - 1;
        unsym_coords[ unsym_actual_nnz ].y = ic - 1;
        unsym_coords[ unsym_actual_nnz++ ].val = val;

        if( unsym_coords[ unsym_actual_nnz - 1 ].x != unsym_coords[ unsym_actual_nnz - 1 ].y )
        {
            unsym_coords[ unsym_actual_nnz ].x = unsym_coords[ unsym_actual_nnz - 1 ].y;
            unsym_coords[ unsym_actual_nnz ].y = unsym_coords[ unsym_actual_nnz - 1 ].x;
            unsym_coords[ unsym_actual_nnz ].val = unsym_coords[ unsym_actual_nnz - 1 ].val;
            unsym_actual_nnz++;
        }
    }
    else
    {
        unsym_coords[ unsym_actual_nnz ].x = ir - 1;
        unsym_coords[ unsym_actual_nnz ].y = ic - 1;
        unsym_coords[ unsym_actual_nnz++ ].val = val;
    }
}

template<typename FloatType>
void MatrixMarketReader<FloatType>::MMGenerateCOOFromFile( FILE *infile )
{
    int unsym_actual_nnz = 0;
    FloatType val;
    int ir, ic;

    const int exp_zeroes = 0;

    for( int i = 0; i < nNZ; i++ )
    {
        if( mm_is_real( Typecode ) )
        {
            if( typeid( FloatType ) == typeid( float ) )
                fscanf( infile, "%d %d %f\n", &ir, &ic, &val );
            else if( typeid( FloatType ) == typeid( double ) )
                fscanf( infile, "%d %d %lf\n", &ir, &ic, &val );

            if( exp_zeroes == 0 && val == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, val );
        }
        else if( mm_is_integer( Typecode ) )
        {
            if(typeid(FloatType) == typeid(float))
                fscanf(infile, "%d %d %f\n", &ir, &ic, &val);
            else if(typeid(FloatType) == typeid(double))
                fscanf(infile, "%d %d %lf\n", &ir, &ic, &val);

            if( exp_zeroes == 0 && val == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, val );

        }
        else if( mm_is_pattern( Typecode ) )
        {
            fscanf( infile, "%d %d", &ir, &ic );
            val = static_cast<FloatType>( MAX_RAND_VAL * ( rand( ) / ( RAND_MAX + 1.0 ) ) );

            if( exp_zeroes == 0 && val == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, val );
        }
    }
    nNZ = unsym_actual_nnz;
}

template<typename FloatType>
int MatrixMarketReader<FloatType>::MMReadBanner( FILE *infile )
{
    char line[ MM_MAX_LINE_LENGTH ];
    char banner[ MM_MAX_TOKEN_LENGTH ];
    char mtx[ MM_MAX_TOKEN_LENGTH ];
    char crd[ MM_MAX_TOKEN_LENGTH ];
    char data_type[ MM_MAX_TOKEN_LENGTH ];
    char storage_scheme[ MM_MAX_TOKEN_LENGTH ];
    char *p;

    mm_clear_typecode( Typecode );

    if( fgets( line, MM_MAX_LINE_LENGTH, infile ) == NULL )
        return MM_PREMATURE_EOF;

    if( sscanf( line, "%s %s %s %s %s", banner, mtx, crd, data_type,
        storage_scheme ) != 5 )
        return MM_PREMATURE_EOF;

    for( p = mtx; *p != '\0'; *p = tolower( *p ), p++ );  /* convert to lower case */
    for( p = crd; *p != '\0'; *p = tolower( *p ), p++ );
    for( p = data_type; *p != '\0'; *p = tolower( *p ), p++ );
    for( p = storage_scheme; *p != '\0'; *p = tolower( *p ), p++ );

    /* check for banner */
    if( strncmp( banner, MatrixMarketBanner, strlen( MatrixMarketBanner ) ) != 0 )
        return MM_NO_HEADER;

    /* first field should be "mtx" */
    if( strcmp( mtx, MM_MTX_STR ) != 0 )
        return  MM_UNSUPPORTED_TYPE;
    mm_set_matrix( Typecode );

    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */
    if( strcmp( crd, MM_SPARSE_STR ) == 0 )
        mm_set_sparse( Typecode );
    else if( strcmp( crd, MM_DENSE_STR ) == 0 )
        mm_set_dense( Typecode );
    else
        return MM_UNSUPPORTED_TYPE;


    /* third field */

    if( strcmp( data_type, MM_REAL_STR ) == 0 )
        mm_set_real( Typecode );
    else
        if( strcmp( data_type, MM_COMPLEX_STR ) == 0 )
            mm_set_complex( Typecode );
        else
            if( strcmp( data_type, MM_PATTERN_STR ) == 0 )
                mm_set_pattern( Typecode );
            else
                if( strcmp( data_type, MM_INT_STR ) == 0 )
                    mm_set_integer( Typecode );
                else
                    return MM_UNSUPPORTED_TYPE;


    /* fourth field */

    if( strcmp( storage_scheme, MM_GENERAL_STR ) == 0 )
        mm_set_general( Typecode );
    else
        if( strcmp( storage_scheme, MM_SYMM_STR ) == 0 )
            mm_set_symmetric( Typecode );
        else
            if( strcmp( storage_scheme, MM_HERM_STR ) == 0 )
                mm_set_hermitian( Typecode );
            else
                if( strcmp( storage_scheme, MM_SKEW_STR ) == 0 )
                    mm_set_skew( Typecode );
                else
                    return MM_UNSUPPORTED_TYPE;

    return 0;

}

template<typename FloatType>
int MatrixMarketReader<FloatType>::MMReadMtxCrdSize( FILE *infile )
{
    char line[ MM_MAX_LINE_LENGTH ];
    int num_items_read;

    /* now continue scanning until you reach the end-of-comments */
    do
    {
        if( fgets( line, MM_MAX_LINE_LENGTH, infile ) == NULL )
            return MM_PREMATURE_EOF;
    } while( line[ 0 ] == '%' );

    /* line[] is either blank or has M,N, nz */
    if( sscanf( line, "%d %d %d", &nRows, &nCols, &nNZ ) == 3 )
        return 0;
    else
        do
        {
            num_items_read = fscanf( infile, "%d %d %d", &nRows, &nCols, &nNZ );
            if( num_items_read == EOF ) return MM_PREMATURE_EOF;
        } while( num_items_read != 3 );

    return 0;
}

// This function reads the file header at the given filepath, and returns the size
// of the sparse matrix in the clsparseCooMatrix parameter.  
// Post-condition: clears clsparseCooMatrix, then sets pCooMatx->m, pCooMatx->n
// pCooMatx->nnz
clsparseStatus
clsparseCooHeaderfromFile( clsparseCooMatrix* cooMatx, const char* filePath )
{
    clsparseCooMatrixPrivate* pCooMatx = static_cast<clsparseCooMatrixPrivate*>( cooMatx );

    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return clsparseInvalidFileFormat;
    }
    else
        return clsparseInvalidFileFormat;

    MatrixMarketReader< cl_float > mm_reader;

    if( mm_reader.MMReadHeader( filePath ) )
        return clsparseInvalidFile;

    clsparseInitCooMatrix( cooMatx );
    pCooMatx->m = mm_reader.GetNumRows( );
    pCooMatx->n = mm_reader.GetNumCols( );
    pCooMatx->nnz = mm_reader.GetNumNonZeroes( );

    return clsparseSuccess;
}

// This function reads the file at the given filepath, and returns the sparse
// matrix in the COO struct.  All matrix data is written to device memory
// Pre-condition: This function assumes that the device memory buffers have been 
// pre-allocated by the caller
clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, clsparseControl control )
{
    clsparseCooMatrixPrivate* pCooMatx = static_cast<clsparseCooMatrixPrivate*>( cooMatx );

    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return clsparseInvalidFileFormat;
    }
    else
        return clsparseInvalidFileFormat;

    MatrixMarketReader< cl_float > mm_reader;
    if( mm_reader.MMReadFormat( filePath ) )
        return clsparseInvalidFile;

    pCooMatx->m = mm_reader.GetNumRows( );
    pCooMatx->n = mm_reader.GetNumCols( );
    pCooMatx->nnz = mm_reader.GetNumNonZeroes( );

    // Transfers data from CPU buffer to GPU buffers
    clMapMemRIAA< cl_float > rCooValues( control->queue( ), pCooMatx->values );
    clMapMemRIAA< cl_int > rCooColIndices( control->queue( ), pCooMatx->colIndices );
    clMapMemRIAA< cl_int > rCooRowIndices( control->queue( ), pCooMatx->rowIndices );

    cl_float* fCooValues = rCooValues.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->valOffset( ), pCooMatx->nnz );
    cl_int* iCooColIndices = rCooColIndices.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->colIndOffset( ), pCooMatx->nnz );
    cl_int* iCooRowIndices = rCooRowIndices.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->rowOffOffset( ), pCooMatx->nnz );

    Coordinate< cl_float >* coords = mm_reader.GetUnsymCoordinates( );
    for( cl_int c = 0; c < pCooMatx->nnz; ++c )
    {
        iCooRowIndices[ c ] = coords[ c ].x;
        iCooColIndices[ c ] = coords[ c ].y;
        fCooValues[ c ] = coords[ c ].val;
    }

    return clsparseSuccess;
}

clsparseStatus
clsparseCooMatrixfromFile( clsparseCooMatrix* cooMatx, const char* filePath, const clAllocator& clAlloc )
{
    clsparseCooMatrixPrivate* pCooMatx = static_cast<clsparseCooMatrixPrivate*>( cooMatx );

    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return clsparseInvalidFileFormat;
    }
    else
        return clsparseInvalidFileFormat;

    // Read data from a file on disk into CPU buffers
    MatrixMarketReader< cl_float > mm_reader;
    if( mm_reader.MMReadFormat( filePath ) )
        return clsparseInvalidFile;

    // BUG: We need to check to see if openCL buffers currently exist and deallocate them first!

    pCooMatx->m = mm_reader.GetNumRows( );
    pCooMatx->n = mm_reader.GetNumCols( );
    pCooMatx->nnz = mm_reader.GetNumNonZeroes( );

#if( BUILD_CLVERSION < 200 )
    pCooMatx->values = static_cast< cl_mem >( clAlloc( pCooMatx->nnz * sizeof( cl_float ) ) );
    pCooMatx->colIndices = static_cast< cl_mem >( clAlloc( pCooMatx->nnz * sizeof( cl_int ) ) );
    pCooMatx->rowIndices = static_cast< cl_mem >( clAlloc( pCooMatx->nnz * sizeof( cl_int ) ) );
#else
    pCooMatx->values = clAlloc( pCooMatx->nnz * sizeof( cl_int ) );
    pCooMatx->colIndices = clAlloc( pCooMatx->nnz * sizeof( cl_float ) );
    pCooMatx->rowIndices = clAlloc( pCooMatx->nnz * sizeof( cl_int ) );
#endif

    // Transfers data from CPU buffer to GPU buffers
    clMapMemRIAA< cl_float > rCooValues( clAlloc.queue, pCooMatx->values );
    clMapMemRIAA< cl_int > rCooColIndices( clAlloc.queue, pCooMatx->colIndices );
    clMapMemRIAA< cl_int > rCooRowIndices( clAlloc.queue, pCooMatx->rowIndices );

    cl_float* fCooValues = rCooValues.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->valOffset( ), pCooMatx->nnz );
    cl_int* iCooColIndices = rCooColIndices.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->colIndOffset( ), pCooMatx->nnz );
    cl_int* iCooRowIndices = rCooRowIndices.clMapMem( CL_TRUE, CL_MAP_WRITE, pCooMatx->rowOffOffset( ), pCooMatx->nnz );

    Coordinate< cl_float >* coords = mm_reader.GetUnsymCoordinates( );
    for( cl_int c = 0; c < pCooMatx->nnz; ++c )
    {
        iCooRowIndices[ c ] = coords[ c ].x;
        iCooColIndices[ c ] = coords[ c ].y;
        fCooValues[ c ] = coords[ c ].val;
    }

    return clsparseSuccess;
}
