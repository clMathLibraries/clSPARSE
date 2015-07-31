#pragma once
#ifndef _CL_COMPUTEROWBLOCKS_H_
#define _CL_COMPUTEROWBLOCKS_H_

#include <iterator>
#include <cassert>

// The row blocks buffer holds a packed set of information used to inform each
// workgroup about how to do its work:
//
// |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
// |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
// |------------Row Information------------|----flag^|---WG ID within a long row---|
//
// The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
// row it will be working on. When one workgroup calculates multiple rows, this
// rowBlock entry and the next one tell it the range of rows to work on.
// The lower 24 bits are used whenever multiple workgroups calculate a single long
// row. This tells each workgroup its ID within that row, so it knows which
// part of the row to operate on.
// Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
// know when the first workgroup for that row has finished initializing the output
// value. While this bit is the same as the first workgroup's flag bit, this
// workgroup will spin-loop.

//  rowBlockType is currently instantiated as ulong
template< typename rowBlockType >
void ComputeRowBlocks( rowBlockType* rowBlocks, size_t& rowBlockSize, const int* rowDelimiters, int nRows, int blkSize )
{
    rowBlockType* rowBlocksBase = rowBlocks;

    *rowBlocks = 0;
    rowBlocks++;
    rowBlockType sum = 0;
    rowBlockType i, last_i = 0;

    // Check to ensure nRows can fit in 32 bits
    if( (rowBlockType)nRows > (rowBlockType)pow( 2, ROW_BITS ) )
    {
        printf( "Number of Rows in the Sparse Matrix is greater than what is supported at present (%d bits) !", ROW_BITS );
        return;
    }

    for( i = 1; i <= nRows; i++ )
    {
        sum += ( rowDelimiters[ i ] - rowDelimiters[ i - 1 ] );

        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WG_BITS == 0
        if( ( i - last_i > 1 ) && sum > blkSize )
        {
            *rowBlocks = ( (i - 1) << (64 - ROW_BITS) );
            rowBlocks++;
            i--;
            last_i = i;
            sum = 0;
        }

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WG_BITS == workgroup ID
        else if( ( i - last_i == 1 ) && sum > blkSize )
        {
            int numWGReq = static_cast< int >( ceil( (double)sum / blkSize ) );

            // Check to ensure #workgroups can fit in WG_BITS bits, if not
            // then the last workgroup will do all the remaining work
            numWGReq = ( numWGReq < (int)pow( 2, WG_BITS ) ) ? numWGReq : (int)pow( 2, WG_BITS );

            for( int w = 1; w < numWGReq; w++ )
            {
                *rowBlocks = ( (i - 1) << (64 - ROW_BITS) );
                *rowBlocks |= static_cast< rowBlockType >( w );
                rowBlocks++;
            }

            *rowBlocks = ( i << (64 -ROW_BITS) );
            rowBlocks++;

            last_i = i;
            sum = 0;
        }
        // sum of non-zero elements is exactly equal to blockSize
        // This is csr-stream case; bottom WG_BITS == 0
        else if( sum == blkSize )
        {
            *rowBlocks = ( i << (64 - ROW_BITS) );
            rowBlocks++;
            last_i = i;
            sum = 0;
        }

    }

    if ((*(rowBlocks-1) >> (64 - ROW_BITS)) != static_cast< rowBlockType>( nRows))
    {
        *rowBlocks = ( static_cast< rowBlockType >( nRows ) << (64 - ROW_BITS) );
        rowBlocks++;
    }

    size_t dist = std::distance( rowBlocksBase, rowBlocks );
    assert( (2 * dist) <= rowBlockSize );

    //   Update the size of rowBlocks to reflect the actual amount of memory used
    // We're multiplying the size by two because the extended precision form of
    // CSR-Adaptive requires more space for the final global reduction.
    rowBlockSize = 2 * dist;
}

#endif
