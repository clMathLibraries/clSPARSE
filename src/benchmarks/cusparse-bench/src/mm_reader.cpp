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
#include <sstream>
#include <cstdio>
#include <iostream>
#include <typeinfo>
#include <vector>
#include <algorithm>

#include "include/external/mmio.h"
#include "include/cufunc_sparse-xx.h"

// Class declarations
template<typename FloatType>
struct Coordinate
{
    clsparseIdx_t x;
    clsparseIdx_t y;
    FloatType val;
};

template <typename FloatType>
class MatrixMarketReader
{
    char Typecode[ 4 ];
    clsparseIdx_t nNZ;
    clsparseIdx_t nRows;
    clsparseIdx_t nCols;
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
    bool MMReadFormat( const std::string& _filename, bool explicit_zeroes );
    int MMReadBanner( FILE* infile );
    int MMReadMtxCrdSize( FILE* infile );
    void MMGenerateCOOFromFile( FILE* infile, bool explicit_zeroes );

    clsparseIdx_t GetNumRows( )
    {
        return nRows;
    }

    clsparseIdx_t GetNumCols( )
    {
        return nCols;
    }

    clsparseIdx_t GetNumNonZeroes( )
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
        printf( "Handling only coordinate format\n" ); return( 1 );
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

    if ( MMReadHeader( mm_file ) )
    {
        printf ("Matrix not supported !\n");
        return 2;
    }

    // If symmetric MM stored file, double the reported size
    if( mm_is_symmetric( Typecode ) )
        nNZ <<= 1;

    ::fclose( mm_file );

    std::clog << "Matrix: " << filename << " [nRow: " << GetNumRows( ) << "] [nCol: " << GetNumCols( ) << "] [nNZ: " << GetNumNonZeroes( ) << "]" << std::endl;

    return 0;
}

template<typename FloatType>
bool MatrixMarketReader<FloatType>::MMReadFormat( const std::string &filename, bool explicit_zeroes )
{
    FILE *mm_file = ::fopen( filename.c_str( ), "r" );
    if( mm_file == NULL )
    {
        printf( "Cannot Open Matrix-Market File !\n" );
        return 1;
    }

    if ( MMReadHeader( mm_file ) )
    {
        printf ("Matrix not supported !\n");
        return 2;
    }

    if( mm_is_symmetric( Typecode ) )
        unsym_coords = new Coordinate<FloatType>[ 2 * nNZ ];
    else
        unsym_coords = new Coordinate<FloatType>[ nNZ ];

    MMGenerateCOOFromFile( mm_file, explicit_zeroes );
    ::fclose( mm_file );

    return 0;
}

template<typename FloatType>
void FillCoordData( char Typecode[ ],
                    Coordinate<FloatType> *unsym_coords,
                    clsparseIdx_t &unsym_actual_nnz,
                    clsparseIdx_t ir,
                    clsparseIdx_t ic,
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
void MatrixMarketReader<FloatType>::MMGenerateCOOFromFile( FILE *infile, bool explicit_zeroes )
{
    clsparseIdx_t unsym_actual_nnz = 0;
    FloatType val;
    clsparseIdx_t ir, ic;

    const int exp_zeroes = explicit_zeroes;

    //silence warnings from fscanf (-Wunused-result)
    clsparseIdx_t rv = 0;

    for ( clsparseIdx_t i = 0; i < nNZ; i++)
    {
        if( mm_is_real( Typecode ) )
        {
            fscanf(infile, "%" SIZET "u", &ir);
            fscanf(infile, "%" SIZET "u", &ic);

            if (typeid(FloatType) == typeid(float))
                rv = fscanf(infile, "%f\n", (float*)(&val));

            else if( typeid( FloatType ) == typeid( double ) )
              rv = fscanf( infile, "%lf\n", (double*)( &val ) );

            if( exp_zeroes == 0 && val == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, val );
        }
        else if( mm_is_integer( Typecode ) )
        {
            fscanf(infile, "%" SIZET "u", &ir);
            fscanf(infile, "%" SIZET "u", &ic);

            if(typeid(FloatType) == typeid(float))
               rv = fscanf(infile, "%f\n", (float*)( &val ) );
            else if(typeid(FloatType) == typeid(double))
               rv = fscanf(infile, "%lf\n", (double*)( &val ) );

            if( exp_zeroes == 0 && val == 0 )
                continue;
            else
                FillCoordData( Typecode, unsym_coords, unsym_actual_nnz, ir, ic, val );

        }
        else if( mm_is_pattern( Typecode ) )
        {
            rv = fscanf(infile, "%" SIZET "u", &ir);
            rv = fscanf(infile, "%" SIZET "u", &ic);

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
    std::stringstream s(line);
    nRows = 0;
    nCols = 0;
    nNZ   = 0;    
    s >> nRows >> nCols >> nNZ;
    if (nRows && nCols && nNZ)
        return 0;
    else
        do
        {
            num_items_read = 0; 
            num_items_read += fscanf( infile, "%" SIZET "u", &nRows );
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
            num_items_read += fscanf(infile,  "%" SIZET "u", &nCols);
            if (num_items_read == EOF) return MM_PREMATURE_EOF;
            num_items_read += fscanf(infile,  "%" SIZET "u", &nNZ);
            if( num_items_read == EOF ) return MM_PREMATURE_EOF;
        } while( num_items_read != 3 );

    return 0;
}

template<typename FloatType>
bool CoordinateCompare( const Coordinate<FloatType> &c1, const Coordinate<FloatType> &c2 )
{
    if( c1.x != c2.x )
        return ( c1.x < c2.x );
    else
        return ( c1.y < c2.y );
}

// This function reads the file at the given filepath, and returns the sparse
// matrix in the COO struct.
template< typename T > int
cooMatrixfromFile( std::vector< clsparseIdx_t >& row_indices, std::vector< clsparseIdx_t >& col_indices,
    std::vector< T >& values, const char* filePath, bool read_explicit_zeroes )
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return 1;
    }
    else
        return 1;

    MatrixMarketReader< T > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return 2;

    clsparseIdx_t m = mm_reader.GetNumRows( );
    clsparseIdx_t n = mm_reader.GetNumCols( );
    clsparseIdx_t nnz = mm_reader.GetNumNonZeroes( );

    row_indices.clear( );
    col_indices.clear( );
    values.clear( );
    row_indices.reserve( m );
    col_indices.reserve( n );
    values.reserve( nnz );

    Coordinate< T >* coords = mm_reader.GetUnsymCoordinates( );
    for( clsparseIdx_t c = 0; c < nnz; ++c )
    {
        row_indices.push_back( coords[ c ].x );
        col_indices.push_back( coords[ c ].y );
        values.push_back( coords[ c ].val );
    }

    return 0;
}

// Explicit template instantiations for float, double
template int cooMatrixfromFile<>( std::vector< clsparseIdx_t >& row_indices, std::vector< clsparseIdx_t >& col_indices,
                                  std::vector< float >& values, const char* filePath, bool read_explicit_zeroes );
template int cooMatrixfromFile<>( std::vector< clsparseIdx_t >& row_indices, std::vector< clsparseIdx_t >& col_indices,
                                  std::vector< double >& values, const char* filePath, bool read_explicit_zeroes );

// This function reads the file at the given filepath, and returns the sparse
// matrix in the CSR struct.
template< typename T > int
csrMatrixfromFile( std::vector< clsparseIdx_t >& row_offsets, std::vector< clsparseIdx_t >& col_indices,
std::vector< T >& values, const char* filePath, bool read_explicit_zeroes )
{
    // Check that the file format is matrix market; the only format we can read right now
    // This is not a complete solution, and fails for directories with file names etc...
    // TODO: Should we use boost filesystem?
    std::string strPath( filePath );
    if( strPath.find_last_of( '.' ) != std::string::npos )
    {
        std::string ext = strPath.substr( strPath.find_last_of( '.' ) + 1 );
        if( ext != "mtx" )
            return 1;
    }
    else
        return 1;

    MatrixMarketReader< T > mm_reader;
    if( mm_reader.MMReadFormat( filePath, read_explicit_zeroes ) )
        return 2;

    clsparseIdx_t m = mm_reader.GetNumRows( );
    clsparseIdx_t n = mm_reader.GetNumCols( );
    clsparseIdx_t nnz = mm_reader.GetNumNonZeroes( );

    row_offsets.clear( );
    col_indices.clear( );
    values.clear( );
    row_offsets.reserve( m );
    col_indices.reserve( n );
    values.reserve( nnz );

    Coordinate< T >* coords = mm_reader.GetUnsymCoordinates( );

    std::sort( coords, coords + nnz, CoordinateCompare< T > );

    clsparseIdx_t current_row = 1;
    row_offsets.push_back( 0 );
    for( clsparseIdx_t i = 0; i < nnz; i++ )
    {
        col_indices.push_back( coords[ i ].y );
        values.push_back( coords[ i ].val );

        if( coords[ i ].x >= current_row )
        {
            row_offsets.push_back( i );
            ++current_row;
        }
    }
    row_offsets.push_back( nnz );

    return 0;
}// end

// Explicit template instantiations for float, double
template int csrMatrixfromFile<>( std::vector< clsparseIdx_t >& row_offsets, std::vector< clsparseIdx_t >& col_indices,
                                  std::vector< float >& values, const char* filePath, bool read_explicit_zeroes );
template int csrMatrixfromFile<>( std::vector< clsparseIdx_t >& row_offsets, std::vector< clsparseIdx_t >& col_indices,
                                  std::vector< double >& values, const char* filePath, bool read_explicit_zeroes );

// This function reads the file header at the given filepath, and gets the
// matrix dimensions
int sparseHeaderfromFile(clsparseIdx_t* nnz, clsparseIdx_t* rows, clsparseIdx_t* cols, const char* filePath)
{
    std::string strPath(filePath);
    if (strPath.find_last_of(".") != std::string::npos)
    {
        std::string ext = strPath.substr(strPath.find('.') + 1);
        if (ext != "mtx")
            return 1;
    }
    else
        return 1;
    MatrixMarketReader< float > mm_reader;
    if (mm_reader.MMReadHeader(filePath))
    {
        return 2;
    }

    *nnz  = mm_reader.GetNumNonZeroes();
    *rows = mm_reader.GetNumRows();
    *cols = mm_reader.GetNumCols();

    return 0; // success
}
