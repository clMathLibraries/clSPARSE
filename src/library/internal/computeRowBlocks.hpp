#pragma once
#ifndef _CL_COMPUTEROWBLOCKS_H_
#define _CL_COMPUTEROWBLOCKS_H_

#include <iterator>
#include <cassert>

#ifndef __has_builtin
  #define __has_builtin(x) 0
#endif

static inline unsigned int flp2(unsigned int x)
{
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    return x - (x >> 1);
}

// Short rows in CSR-Adaptive are batched together into a single row block.
// If there are a relatively small number of these, then we choose to do
// a horizontal reduction (groups of threads all reduce the same row).
// If there are many threads (e.g. more threads than the maximum size
// of our workgroup) then we choose to have each thread serially reduce
// the row.
// This function calculates the number of threads that could team up
// to reduce these groups of rows. For instance, if you have a
// workgroup size of 256 and 4 rows, you could have 64 threads
// working on each row. If you have 5 rows, only 32 threads could
// reliably work on each row because our reduction assumes power-of-2.
template< typename rowBlockType >
static inline rowBlockType numThreadsForReduction(rowBlockType num_rows)
{
#if defined(__INTEL_COMPILER)
    return 256 >> (_bit_scan_reverse(num_rows-1)+1);
#elif (defined(__clang__) && __has_builtin(__builtin_clz)) || \
      !defined(__clang) && \
      defined(__GNUG__) && ((__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__) > 30202)
    return (256 >> (8*sizeof(int)-__builtin_clz(num_rows-1)));
#elif defined(_MSC_VER) && (_MSC_VER >= 1400)
    int bit_returned;
    _BitScanReverse(&bit_returned, (num_rows-1));
    return 256 >> (bit_returned+1);
#else
    return flp2(256/num_rows);
#endif
}

// The row blocks buffer holds a packed set of information used to inform each
// workgroup about how to do its work:
//
// |6666 5555 5555 5544 4444 4444 3333 3333|3322 2222|2222 1111 1111 1100 0000 0000|
// |3210 9876 5432 1098 7654 3210 9876 5432|1098 7654|3210 9876 5432 1098 7654 3210|
// |------------Row Information------------|--------^|---WG ID within a long row---|
// |                                       |    flag/|or # reduce threads for short|
//
// The upper 32 bits of each rowBlock entry tell the workgroup the ID of the first
// row it will be working on. When one workgroup calculates multiple rows, this
// rowBlock entry and the next one tell it the range of rows to work on.
// The lower 24 bits are used whenever multiple workgroups calculate a single long
// row. This tells each workgroup its ID within that row, so it knows which
// part of the row to operate on.
// Alternately, on "short" row blocks, the lower bits are used to communicate
// the number of threads that should be used for the reduction. Pre-calculating
// this on the CPU-side results in a noticable performance uplift on many matrices.
// Bit 24 is a flag bit used so that the multiple WGs calculating a long row can
// know when the first workgroup for that row has finished initializing the output
// value. While this bit is the same as the first workgroup's flag bit, this
// workgroup will spin-loop.

//  rowBlockType is currently instantiated as ulong
template< typename rowBlockType >
void ComputeRowBlocks( rowBlockType* rowBlocks, size_t& rowBlockSize, const int* rowDelimiters, int nRows, int blkSize, int blkMultiplier )
{
    rowBlockType* rowBlocksBase;

    rowBlocksBase = rowBlocks;
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
        int row_length = ( rowDelimiters[ i ] - rowDelimiters[ i - 1 ] );
        sum += row_length;

        // exactly one row results in non-zero elements to be greater than blockSize
        // This is csr-vector case; bottom WG_BITS == workgroup ID
        if( ( i - last_i == 1 ) && sum > blkSize )
        {
            int numWGReq = static_cast< int >( ceil( (double)sum / (blkMultiplier*blkSize) ) );

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
        // more than one row results in non-zero elements to be greater than blockSize
        // This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
        else if( ( i - last_i > 1 ) && sum > blkSize )
        {
            i--; // This row won't fit, so back off one.
            *rowBlocks = ( i << (64 - ROW_BITS) );
            // Fill in the low-order bits with the numThreadsForRed
            if ((i - last_i) > 2)
                *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
            rowBlocks++;
            last_i = i;
            sum = 0;
        }
        // This is csr-stream case; bottom WG_BITS = number of parallel reduction threads
        else if( sum == blkSize )
        {
            *rowBlocks = ( i << (64 - ROW_BITS) );
            // Fill in the low-order bits with the numThreadsForRed
            if ((i - last_i) > 2)
                *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
            rowBlocks++;
            last_i = i;
            sum = 0;
        }

    }

    // If we didn't fill a row block with the last row, make sure we don't lose it.
    if ( (*(rowBlocks-1) >> (64 - ROW_BITS)) != static_cast< rowBlockType>(nRows) )
    {
        *rowBlocks = ( static_cast< rowBlockType >( nRows ) << (64 - ROW_BITS) );
        // Fill in the low-order bits with the numThreadsForRed
        if ((nRows - last_i) > 2)
            *(rowBlocks-1) |= numThreadsForReduction(i - last_i);
        rowBlocks++;
    }

    size_t dist = std::distance( rowBlocksBase, rowBlocks );
    assert( (2 * dist) <= rowBlockSize );
    // Update the size of rowBlocks to reflect the actual amount of memory used
    // We're multiplying the size by two because the extended precision form of
    // CSR-Adaptive requires more space for the final global reduction.
    rowBlockSize = 2 * dist;
}

#endif
