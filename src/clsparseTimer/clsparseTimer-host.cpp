/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/

// StatTimer.cpp : Defines the exported functions for the DLL application.
//

#include <cassert>
#include <iostream>
#include <iomanip>
#include <string>
#include <functional>
#include <cmath>
#include <limits>

#include "clsparseTimer-host.hpp"

#if defined( _WIN32 )
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/time.h>
#endif

//	Format an unsigned number with comma thousands separator
//
namespace {
    std::string commatize( cl_ulong number )
    {
        static char scratch[ 8 * sizeof( cl_ulong ) ];

        char* ptr = scratch + sizeof( scratch );
        *( --ptr ) = 0;

        for( int digits = 3;; )
        {
            *( --ptr ) = '0' + int( number % 10 );
            number /= 10;
            if( 0 == number )
                break;
            if( --digits <= 0 )
            {
                *( --ptr ) = ',';
                digits = 3;
            }
        }

        return std::string{ ptr };
    }

    //	Functor object to help with accumulating values in vectors
    template< typename T >
    struct Accumulator: public std::unary_function < T, void >
    {
        T acc;

        Accumulator( ): acc( 0 ) {}
        void operator( )( T x ) { acc += x; }
    };

    //	Unary predicate used for remove_if() algorithm
    //	Currently, RangeType is expected to be a floating point type, and ValType an integer type
    template< typename RangeType, typename ValType >
    struct PruneRange
    {
        RangeType lower, upper;

        PruneRange( RangeType mean, RangeType stdev ): lower( mean - stdev ), upper( mean + stdev ) {}

        bool operator( )( ValType val )
        {
            //	These comparisons can be susceptible to signed/unsigned casting problems
            //	This is why we cast ValType to RangeType, because RangeType should always be floating and signed
            if( static_cast<RangeType>( val ) < lower )
                return true;
            else if( static_cast<RangeType>( val ) > upper )
                return true;

            return false;
        }
    };
}
clsparseHostTimer&
clsparseHostTimer::getInstance( )
{
    static	clsparseHostTimer	timer;
    return	timer;
}

clsparseHostTimer::clsparseHostTimer( ): nEvents( 0 ), nSamples( 0 ), normalize( true )
{
#if defined( _WIN32 )
    //	OS call to get ticks per second2
    ::QueryPerformanceFrequency( reinterpret_cast<LARGE_INTEGER*>( &clkFrequency ) );
#else
    res.tv_sec	= 0;
    res.tv_nsec	= 0;
    clkFrequency 	= 0;

    //	clock_getres() return 0 for success
    //	If the function fails (monotonic clock not supported), we default to a lower resolution timer
    //	if( ::clock_getres( CLOCK_MONOTONIC, &res ) )
    {
        clkFrequency = 1000000;
    }
    //	else
    //	{
    //	    // Turn time into frequency
    //		clkFrequency = res.tv_nsec * 1000000000;
    //	}

#endif
}

clsparseHostTimer::~clsparseHostTimer( )
{}

void
clsparseHostTimer::Clear( )
{
    labelID.clear( );
    clkStart.clear( );
    clkTicks.clear( );
}

void
clsparseHostTimer::Reset( )
{
    if( nEvents == 0 || nSamples == 0 )
        throw	std::runtime_error( "StatisticalTimer::Reserve( ) was not called before Reset( )" );

    clkStart.clear( );
    clkTicks.clear( );

    clkStart.resize( nEvents );
    clkTicks.resize( nEvents );

    for( cl_uint i = 0; i < nEvents; ++i )
    {
        clkTicks.at( i ).reserve( nSamples );
    }

    return;
}

//	The caller can pre-allocate memory, to improve performance.
//	nEvents is an approximate value for how many seperate events the caller will think
//	they will need, and nSamples is a hint on how many samples we think we will take
//	per event
void
clsparseHostTimer::Reserve( size_t nEvents, size_t nSamples )
{
    this->nEvents = std::max< size_t >( 1, nEvents );
    this->nSamples = std::max< size_t >( 1, nSamples );

    Clear( );
    labelID.reserve( nEvents );

    clkStart.resize( nEvents );
    clkTicks.resize( nEvents );

    for( cl_uint i = 0; i < nEvents; ++i )
    {
        clkTicks.at( i ).reserve( nSamples );
    }
}

void
clsparseHostTimer::setNormalize( bool norm )
{
    normalize = norm;
}

void
clsparseHostTimer::Start( size_t id )
{
#if defined( _WIN32 )
    ::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &clkStart.at( id ) ) );
#else
    if( clkFrequency )
    {
        struct timeval s;
        gettimeofday( &s, 0 );
        clkStart.at( id ) = (cl_ulong)s.tv_sec * 1000000 + (cl_ulong)s.tv_usec;
    }
    else
    {

    }
#endif
}

void
clsparseHostTimer::Stop( size_t id )
{
    cl_ulong n;

#if defined( _WIN32 )
    ::QueryPerformanceCounter( reinterpret_cast<LARGE_INTEGER*>( &n ) );
#else
    struct timeval s;
    gettimeofday( &s, 0 );
    n = (cl_ulong)s.tv_sec * 1000000 + (cl_ulong)s.tv_usec;
#endif

    n -= clkStart.at( id );
    clkStart.at( id ) = 0;
    AddSample( id, n );
}

void
clsparseHostTimer::AddSample( const size_t id, const cl_ulong n )
{
    clkTicks.at( id ).push_back( n );
}

//	This function's purpose is to provide a mapping from a 'friendly' human readable text string
//	to an index into internal data structures.
size_t
clsparseHostTimer::getUniqueID( const std::string& label, cl_uint groupID )
{
    //	I expect labelID will hardly ever grow beyond 30, so it's not of any use
    //	to keep this sorted and do a binary search

    labelPair	sItem = std::make_pair( label, groupID );

    stringVector::iterator	iter;
    iter = std::find( labelID.begin( ), labelID.end( ), sItem );

    if( iter != labelID.end( ) )
        return	std::distance( labelID.begin( ), iter );

    labelID.push_back( sItem );

    return	labelID.size( ) - 1;

}

cl_double
clsparseHostTimer::getMean( size_t id ) const
{
    if( clkTicks.empty( ) )
        return	0;

    size_t	N = clkTicks.at( id ).size( );

    Accumulator<cl_ulong> sum = std::for_each( clkTicks.at( id ).begin( ), clkTicks.at( id ).end( ), Accumulator<cl_ulong>( ) );

    return	static_cast<cl_double>( sum.acc ) / N;
}

cl_double
clsparseHostTimer::getVariance( size_t id ) const
{
    if( clkTicks.empty( ) )
        return	0;

    cl_double	mean = getMean( id );

    size_t	N = clkTicks.at( id ).size( );
    cl_double	sum = 0;

    for( cl_uint i = 0; i < N; ++i )
    {
        cl_double	diff = clkTicks.at( id ).at( i ) - mean;
        diff *= diff;
        sum += diff;
    }

    return	 sum / N;
}

cl_double
clsparseHostTimer::getStdDev( size_t id ) const
{
    cl_double	variance = getVariance( id );

    return	sqrt( variance );
}

cl_double
clsparseHostTimer::getAverageTime( size_t id ) const
{
    if( normalize )
        return getMean( id ) / clkFrequency;
    else
        return getMean( id );
}

cl_double
clsparseHostTimer::getMinimumTime( size_t id ) const
{
    clkVector::const_iterator iter = std::min_element( clkTicks.at( id ).begin( ), clkTicks.at( id ).end( ) );

    if( iter != clkTicks.at( id ).end( ) )
    {
        if( normalize )
            return static_cast<cl_double>( *iter ) / clkFrequency;
        else
            return static_cast<cl_double>( *iter );
    }
    else
        return	0;
}

size_t
clsparseHostTimer::pruneOutliers( size_t id, cl_double multiple )
{
    if( clkTicks.empty( ) )
        return 0;

    cl_double mean = getMean( id );
    cl_double stdDev = getStdDev( id );

    clkVector& clks = clkTicks.at( id );

    //	Look on p. 379, "The C++ Standard Library"
    //	std::remove_if does not actually erase, it only copies elements, it returns new 'logical' end
    clkVector::iterator	newEnd = std::remove_if( clks.begin( ), clks.end( ), PruneRange< cl_double, cl_ulong >( mean, multiple*stdDev ) );

    clkVector::difference_type dist = std::distance( newEnd, clks.end( ) );

    if( dist != 0 )
        clks.erase( newEnd, clks.end( ) );

    assert( dist < std::numeric_limits< cl_uint >::max( ) );

    return dist;
}

size_t
clsparseHostTimer::pruneOutliers( cl_double multiple )
{
    const int tableWidth = 60;
    const int tableHalf = tableWidth / 2;
    const int tableThird = tableWidth / 3;
    const int tableFourth = tableWidth / 4;
    const int tableFifth = tableWidth / 5;

    //	Print label of timer, in a header
    std::string header( "StdDev" );
    size_t	sizeTitle = ( header.size( ) + 6 ) / 2;

    std::cout << std::endl;
    std::cout << std::setfill( '=' ) << std::setw( tableHalf ) << header << " ( " << multiple << " )"
        << std::setw( tableHalf - sizeTitle ) << "=" << std::endl;
    std::cout << std::setfill( ' ' );

    size_t tCount = 0;
    for( cl_uint l = 0; l < labelID.size( ); ++l )
    {
        size_t tSamples = clkTicks.at( l ).size( );
        size_t lCount = pruneOutliers( l, multiple );

        std::cout << labelID[ l ].first << "[ 0 ]" << ": Pruning " << lCount << " samples out of " << tSamples << std::endl;
        tCount += lCount;
    }

    return	tCount;
}

void
clsparseHostTimer::Print( cl_ulong flopCount, std::string unit )
{
    const int tableWidth = 60;
    const int tableHalf = tableWidth / 2;
    const int tableThird = tableWidth / 3;
    const int tableFourth = tableWidth / 4;
    const int tableFifth = tableWidth / 5;

    for( cl_uint id = 0; id < labelID.size( ); ++id )
    {
        size_t halfString = labelID[ id ].first.size( ) / 2;

        //	Print label of timer, in a header
        std::cout << std::endl << std::setw( tableHalf + halfString ) << std::setfill( '=' ) << labelID[ id ].first
            << std::setw( tableHalf - halfString ) << "=" << std::endl;
        std::cout << std::setfill( ' ' );

        cl_double time = getAverageTime( id ) * 1e9;
        cl_double gFlops = flopCount / time;

        std::cout << std::setw( tableFourth ) << unit << ":"
            << std::setw( 2 * tableFourth ) << gFlops << std::endl;
        std::cout << std::setw( tableFourth ) << "Time (ns):"
            << std::setw( 3 * tableFourth ) << commatize( static_cast<cl_ulong>( time ) ) << std::endl;
        std::cout << std::endl;
    }
}

//	Defining an output print operator
std::ostream&
operator<<( std::ostream& os, const clsparseHostTimer& st )
{
    if( st.clkTicks.empty( ) )
        return	os;

    std::ios::fmtflags bckup = os.flags( );

    for( cl_uint l = 0; l < st.labelID.size( ); ++l )
    {
        cl_ulong min = 0;
        clsparseHostTimer::clkVector::const_iterator iter = std::min_element( st.clkTicks.at( l ).begin( ), st.clkTicks.at( l ).end( ) );

        if( iter != st.clkTicks.at( l ).end( ) )
            min = *iter;

        os << st.labelID[ l ].first << ", " << st.labelID[ l ].second << std::fixed << std::endl;
        os << "Min:," << min << std::endl;
        os << "Mean:," << st.getMean( l ) << std::endl;
        os << "StdDev:," << st.getStdDev( l ) << std::endl;
        os << "AvgTime:," << st.getAverageTime( l ) << std::endl;
        os << "MinTime:," << st.getMinimumTime( l ) << std::endl;

        //for( cl_uint	t = 0; t < st.clkTicks[l].size( ); ++t )
        //{
        //	os << st.clkTicks[l][t]<< ",";
        //}
        os << "\n" << std::endl;

    }

    os.flags( bckup );

    return	os;
}
