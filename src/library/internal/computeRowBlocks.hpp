#pragma once
#ifndef _CL_COMPUTEROWBLOCKS_H_
#define _CL_COMPUTEROWBLOCKS_H_

#include <iterator>
#include <cassert>

template< typename rowBlockType >
void ComputeRowBlocks( rowBlockType* rowBlocks, size_t rowBlockSize, const int* rowDelimiters, int nRows, int blkSize )
{
    rowBlockType* rowBlocksBase = rowBlocks;

    *rowBlocks = 0;
    rowBlocks++;
    rowBlockType sum = 0;
    rowBlockType i, last_i = 0;

    // Check to ensure nRows can fit in 32 bits
    if( (rowBlockType)nRows > (rowBlockType)pow( 2, ( 64 - ROW_BITS ) ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present ((64-ROW_BITS) bits) !" );
        exit( 0 );
    }

    for( i = 1; i <= nRows; i++ )
    {
        sum += ( rowDelimiters[ i ] - rowDelimiters[ i - 1 ] );

        // more than one row results in non-zero elements
        // to be greater than blockSize
        if( ( i - last_i > 1 ) && sum > blkSize )
        {
            *rowBlocks = ( (i - 1) << ROW_BITS );
            rowBlocks++;
            i--;
            last_i = i;
            sum = 0;
        }

        // exactly one row results in non-zero elements
        // to be greater than blockSize
        else if( ( i - last_i == 1 ) && sum > blkSize )
        {
            int numWGReq = static_cast< int >( ceil( (double)sum / blkSize ) );

            // Check to ensure #workgroups can fit in 24 bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = ( numWGReq < (int)pow( 2, WG_BITS ) ) ? numWGReq : (int)pow( 2, WG_BITS );

            for( int w = 1; w < numWGReq; w++ )
            {
                *rowBlocks = ( (i - 1) << ROW_BITS );
                *rowBlocks |= static_cast< rowBlockType >( w );
                rowBlocks++;
            }

            *rowBlocks = ( i << ROW_BITS );
            rowBlocks++;

            last_i = i;
            sum = 0;
        }
        // sum of non-zero elements is exactly
        // equal to blockSize
        else if( sum == blkSize )
        {
            *rowBlocks = ( i << ROW_BITS );
            rowBlocks++;
            last_i = i;
            sum = 0;
        }

    }

    *rowBlocks = ( static_cast< rowBlockType >( nRows ) << ROW_BITS );
    rowBlocks++;

    size_t dist = std::distance( rowBlocksBase, rowBlocks );
    assert( dist < rowBlockSize );
}

#endif