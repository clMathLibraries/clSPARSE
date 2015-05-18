/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/

// clsparseTimer.device.cpp : Defines the exported functions for the DLL application.
//

#include <cassert>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <functional>
#include <cmath>
#include "clsparseTimer.device.hpp"

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

        Accumulator( ): acc( 0 )
        {
        }
        void operator( )( T x )
        {
            acc += x;
        }
    };

    //	Functor object to help with accumulating values in vectors
    template< >
    struct Accumulator < StatData >
    {
        StatData acc;

        Accumulator( )
        {
        }
        void operator( )( const StatData& x )
        {
            acc.deltaNanoSec += x.deltaNanoSec;
        }
    };
}


//	Unary predicate used for remove_if() algorithm
//	Currently, RangeType is expected to be a floating point type, and ValType an integer type
template< typename T, typename R >
struct PruneRange: public std::binary_function < T, R, bool >
{
    R lower, upper;

    PruneRange( R mean, R stdev ): lower( mean - stdev ), upper( mean + stdev )
    {
    }

    bool operator( )( T val )
    {
        //	These comparisons can be susceptible to signed/unsigned casting problems
        //	This is why we cast ValType to RangeType, because RangeType should always be floating and signed
        if( static_cast<R>( val ) < lower )
            return true;
        else if( static_cast<R>( val ) > upper )
            return true;

        return false;
    }
};

//	Template specialization for StatData datatypes
template< >
struct PruneRange < StatData, cl_double >
{
    StatData mean;
    cl_double stdDev;

    PruneRange( StatData m, cl_double s ): mean( m ), stdDev( s )
    {
    }

    bool operator( )( StatData val )
    {
        //	These comparisons can be susceptible to signed/unsigned casting problems
        //	This is why we cast ValType to RangeType, because RangeType should always be floating and signed
        if( val.doubleNanoSec < ( mean.doubleNanoSec - stdDev ) )
            return true;
        else if( val.doubleNanoSec > ( mean.doubleNanoSec + stdDev ) )
            return true;

        return false;
    }
};

//	Sorting operator for struct StatData, such that it can be used in a map
bool operator<( const StatData& lhs, const StatData& rhs )
{
    if( lhs.deltaNanoSec < rhs.deltaNanoSec )
        return true;
    else
        return false;
}

clsparseDeviceTimer&
clsparseDeviceTimer::getInstance( )
{
    static	clsparseDeviceTimer	timer;
    return	timer;
}

clsparseDeviceTimer::clsparseDeviceTimer( ): nEvents( 0 ), nSamples( 0 ), currID( 0 ), currSample( 0 ), currRecord( 0 )
{
}

clsparseDeviceTimer::~clsparseDeviceTimer( )
{
}

void
clsparseDeviceTimer::Clear( )
{
    labelID.clear( );
    timerData.clear( );

    nEvents = 0;
    nSamples = 0;
    currID = 0;
    currSample = 0;
    currRecord = 0;
}

//	The caller can pre-allocate memory, to improve performance.
//	nEvents is an approximate value for how many seperate events the caller will think
//	they will need, and nSamples is a hint on how many samples we think we will take
//	per event
void
clsparseDeviceTimer::Reserve( size_t nE, size_t nS )
{
    Clear( );
    nEvents = std::max< size_t >( 1, nE );
    nSamples = std::max< size_t >( 1, nS );

    labelID.reserve( nEvents );
    timerData.resize( nEvents );
}

void
clsparseDeviceTimer::Reset( )
{
    if( nEvents == 0 || nSamples == 0 )
        throw	std::runtime_error( "StatisticalTimer::Reserve( ) was not called before Reset( )" );

    Reserve( nEvents, nSamples );

    return;
}

void
clsparseDeviceTimer::setNormalize( bool norm )
{
}

void
clsparseDeviceTimer::Start( size_t id )
{
    currID = id;
    currSample = 0;
}

void
clsparseDeviceTimer::Stop( size_t id )
{
    ++currRecord;
}

void
clsparseDeviceTimer::AddSample( std::vector< cl::Event > vecEvent )
{
    if( timerData.empty( ) )
        return;

    if( currRecord == 0 )
    {
        timerData.at( currID ).push_back( StatDataVec( ) );
        timerData.at( currID ).back( ).reserve( nSamples );
        timerData.at( currID ).back( ).push_back( StatData( vecEvent ) );
    }
    else
    {
        timerData.at( currID ).at( currSample ).push_back( StatData( vecEvent ) );
        ++currSample;
    }
}

//	This function's purpose is to provide a mapping from a 'friendly' human readable text string
//	to an index into internal data structures.
size_t
clsparseDeviceTimer::getUniqueID( const std::string& label, cl_uint groupID )
{
    //	I expect labelID will hardly ever grow beyond 30, so it's not of any use
    //	to keep this sorted and do a binary search

    idPair	sItem = std::make_pair( label, groupID );

    idVector::iterator	iter;
    iter = std::find( labelID.begin( ), labelID.end( ), sItem );

    if( iter != labelID.end( ) )
        return	std::distance( labelID.begin( ), iter );

    labelID.push_back( sItem );

    return	labelID.size( ) - 1;

}

void clsparseDeviceTimer::queryOpenCL( size_t id )
{
    for( size_t s = 0; s < timerData.at( id ).size( ); ++s )
    {
        for( size_t n = 0; n < timerData.at( id ).at( s ).size( ); ++n )
        {
            StatData& sd = timerData[ id ][ s ][ n ];

            cl_ulong profStart, profEnd = 0;
            cl_int err = 0;
            sd.deltaNanoSec = 0;

            for( size_t i = 0; i < sd.outEvents.size( ); ++i )
            {
                profStart = sd.outEvents[ i ].getProfilingInfo<CL_PROFILING_COMMAND_START>( &err );
                OPENCL_V_THROW( err, "clsparseDeviceTimer::queryOpenCL" );

                profEnd = sd.outEvents[ i ].getProfilingInfo<CL_PROFILING_COMMAND_END>( &err );
                OPENCL_V_THROW( err, "clsparseDeviceTimer::queryOpenCL" );

                sd.deltaNanoSec += ( profEnd - profStart );
            }

            sd.doubleNanoSec = static_cast<cl_double>( sd.deltaNanoSec );
        }
    }
}

std::vector< StatData >
clsparseDeviceTimer::getMean( size_t id )
{
    //	Prep the data; query openCL for the timer information
    queryOpenCL( id );

    std::vector< StatData > meanVec;
    for( size_t s = 0; s < timerData.at( id ).size( ); ++s )
    {
        Accumulator< StatData > sum = std::for_each( timerData.at( id ).at( s ).begin( ), timerData.at( id ).at( s ).end( ),
            Accumulator< StatData >( ) );

        StatData tmp = timerData[ id ][ s ].front( );
        tmp.doubleNanoSec = static_cast<cl_double>( sum.acc.deltaNanoSec ) / timerData.at( id ).at( s ).size( );
        meanVec.push_back( tmp );
    }

    return meanVec;
}

std::vector< StatData >
clsparseDeviceTimer::getVariance( size_t id )
{
    std::vector< StatData > variance = getMean( id );

    for( cl_uint v = 0; v < variance.size( ); ++v )
    {
        double sum = 0;
        for( cl_uint n = 0; n < timerData[ id ][ v ].size( ); ++n )
        {
            cl_double	diff = static_cast<cl_double>( timerData[ id ][ v ][ n ].deltaNanoSec ) - variance[ v ].doubleNanoSec;
            diff *= diff;
            sum += diff;
        }

        variance[ v ].doubleNanoSec = sum / timerData[ id ][ v ].size( );
    }

    return variance;
}

std::vector< StatData >
clsparseDeviceTimer::getStdDev( size_t id )
{
    std::vector< StatData > stddev = getVariance( id );

    for( cl_uint v = 0; v < stddev.size( ); ++v )
    {
        stddev[ v ].doubleNanoSec = sqrt( stddev[ v ].doubleNanoSec );
    }

    return stddev;
}

std::vector< StatData >
clsparseDeviceTimer::getAverageTime( size_t id )
{
    return getMean( id );
}

std::vector< StatData >
clsparseDeviceTimer::getMinimumTime( size_t id )
{
    //	Prep the data; query openCL for the timer information
    queryOpenCL( id );

    std::vector< StatData > minTime;
    for( size_t s = 0; s < timerData.at( id ).size( ); ++s )
    {
        StatDataVec::iterator iter
            = std::min_element( timerData.at( id ).at( s ).begin( ), timerData.at( id ).at( s ).end( ) );

        if( iter != timerData.at( id ).at( s ).end( ) )
        {
            iter->doubleNanoSec = static_cast<cl_double>( iter->deltaNanoSec ) / timerData.at( id ).at( s ).size( );
            minTime.push_back( *iter );
        }
        else
            return std::vector< StatData >( );
    }

    return minTime;
}

std::vector< size_t >
clsparseDeviceTimer::pruneOutliers( size_t id, cl_double multiple )
{
    std::vector< StatData > mean = getMean( id );
    std::vector< StatData > stdDev = getStdDev( id );

    std::vector< size_t > totalPrune;
    for( size_t s = 0; s < timerData.at( id ).size( ); ++s )
    {
        //	Look on p. 379, "The C++ Standard Library"
        //	std::remove_if does not actually erase, it only copies elements, it returns new 'logical' end
        StatDataVec::iterator newEnd = std::remove_if( timerData.at( id ).at( s ).begin( ), timerData.at( id ).at( s ).end( ),
            PruneRange< StatData, cl_double >( mean[ s ], multiple * stdDev[ s ].doubleNanoSec ) );

        StatDataVec::difference_type dist = std::distance( newEnd, timerData.at( id ).at( s ).end( ) );

        if( dist != 0 )
            timerData.at( id ).at( s ).erase( newEnd, timerData.at( id ).at( s ).end( ) );

        totalPrune.push_back( dist );
    }

    return totalPrune;
}

size_t
clsparseDeviceTimer::pruneOutliers( cl_double multiple )
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
        std::vector< size_t > lCount = pruneOutliers( l, multiple );

        for( cl_uint c = 0; c < lCount.size( ); ++c )
        {
            std::cout << labelID[ l ].first << "[ " << c << " ]" << ": Pruning " << lCount[ c ] << " samples out of " << currRecord << std::endl;
            tCount += lCount[ c ];
        }
    }

    return tCount;
}

void
clsparseDeviceTimer::Print( cl_ulong flopCount, std::string unit )
{
    const int tableWidth = 60;
    const int tableHalf = tableWidth / 2;
    const int tableThird = tableWidth / 3;
    const int tableFourth = tableWidth / 4;
    const int tableFifth = tableWidth / 5;

    for( cl_uint id = 0; id < labelID.size( ); ++id )
    {
        size_t	halfString = labelID[ id ].first.size( ) / 2;

        //	Print label of timer, in a header
        std::cout << std::endl << std::setw( tableHalf + halfString ) << std::setfill( '=' ) << labelID[ id ].first
            << std::setw( tableHalf - halfString ) << "=" << std::endl;
        std::cout << std::setfill( ' ' );

        std::vector< StatData > mean = getMean( id );

        //	Print each individual dimension
        std::stringstream catLengths;
        for( cl_uint t = 0; t < mean.size( ); ++t )
        {
            cl_double time = mean[ t ].doubleNanoSec;
            cl_double gFlops = flopCount / time;

            if( mean[ t ].outEvents.size( ) != 0 )
            {
                std::cout << std::setw( tableFourth ) << "OutEvents:" << std::setw( tableThird );
                for( size_t i = 0; i < mean[ t ].outEvents.size( ); ++i )
                {
                    std::cout << mean[ t ].outEvents[ i ]( );
                    if( i < ( mean[ t ].outEvents.size( ) - 1 ) )
                    {
                        std::cout << "," << std::endl;
                        std::cout << std::setw( tableFourth + tableThird );
                    }
                }
                std::cout << std::endl;
            }


            std::cout << std::setw( tableFourth ) << unit << ":"
                << std::setw( 2 * tableFourth ) << gFlops << std::endl;
            std::cout << std::setw( tableFourth ) << "Time (ns):"
                << std::setw( 3 * tableFourth ) << commatize( static_cast<cl_ulong>( time ) ) << std::endl;
            std::cout << std::endl;
        }
    }
}

//	Defining an output print operator
std::ostream&
operator<<( std::ostream& os, const clsparseDeviceTimer& st )
{
    //if( st.clkTicks.empty( ) )
    //	return	os;

    //std::ios::fmtflags bckup	= os.flags( );

    //for( cl_uint l = 0; l < st.labelID.size( ); ++l )
    //{
    //	cl_ulong min	= 0;
    //	clkVector::const_iterator iter	= std::min_element( st.clkTicks.at( l ).begin( ), st.clkTicks.at( l ).end( ) );

    //	if( iter != st.clkTicks.at( l ).end( ) )
    //		min		= *iter;

    //	os << st.labelID[l].first << ", " << st.labelID[l].second << std::fixed << std::endl;
    //	os << "Min:," << min << std::endl;
    //	os << "Mean:," << st.getMean( l ) << std::endl;
    //	os << "StdDev:," << st.getStdDev( l ) << std::endl;
    //	os << "AvgTime:," << st.getAverageTime( l ) << std::endl;
    //	os << "MinTime:," << st.getMinimumTime( l ) << std::endl;

    //	//for( cl_uint	t = 0; t < st.clkTicks[l].size( ); ++t )
    //	//{
    //	//	os << st.clkTicks[l][t]<< ",";
    //	//}
    //	os << "\n" << std::endl;

    //}

    //os.flags( bckup );

    return	os;
}
