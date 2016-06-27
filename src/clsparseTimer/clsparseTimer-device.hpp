/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef _STATISTICALTIMER_GPU_H_
#define _STATISTICALTIMER_GPU_H_
#include <iosfwd>
#include <vector>
#include <algorithm>
#include <cmath>
#include "clsparseTimer.hpp"

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION BUILD_CLVERSION
#define CL_HPP_TARGET_OPENCL_VERSION BUILD_CLVERSION

#include <CL/cl2.hpp>

/**
 * \file clsparseTimer-device.hpp
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 *	This class is implemented entirely in the header, to facilitate inclusion into multiple
 *	projects without needing to compile an object file for each project.
 */

struct StatData
{
    cl_ulong deltaNanoSec;
    double doubleNanoSec;

    std::vector< cl::Event > outEvents;

    StatData( ): deltaNanoSec( 0 ), doubleNanoSec( 0 )
    {}

    StatData( std::vector< cl::Event >& vecEvent ): deltaNanoSec( 0 ), doubleNanoSec( 0 )
    {
        outEvents.swap( vecEvent );
    }

    double calcFlops( )
    {
        return 0.0;
    }

};

//	Sorting operator for struct StatData, such that it can be used in a map
bool operator<( const StatData& lhs, const StatData& rhs );

class clsparseDeviceTimer: public clsparseTimer
{
    //	Typedefs to handle the data that we store
    typedef std::vector< StatData > StatDataVec;
    typedef std::vector< StatDataVec > PerEnqueueVec;

    //	In order to calculate statistics <std. dev.>, we need to keep a history of our timings
    std::vector< PerEnqueueVec > timerData;

    //	Typedefs to handle the identifiers we use for our timers
    typedef	std::pair< std::string, cl_uint > idPair;
    typedef	std::vector< idPair > idVector;
    idVector labelID;

    //	Between each Start/Stop pair, we need to count how many AddSamples were made.
    size_t currSample, currRecord;

    //	Saved sizes for our vectors, used in Reset() to reallocate vectors
    StatDataVec::size_type	nEvents, nSamples;
    size_t currID;

    /**
     * \fn clsparseDeviceTimer()
     * \brief Constructor for StatisticalTimer that initializes the class
     *	This is private so that user code cannot create their own instantiation.  Instead, you
     *	must go through getInstance( ) to get a reference to the class.
     */
    clsparseDeviceTimer( );

    /**
     * \fn ~clsparseDeviceTimer()
     * \brief Destructor for StatisticalTimer that cleans up the class
     */
    ~clsparseDeviceTimer( );

    /**
     * \fn clsparseDeviceTimer(const StatisticalTimer& )
     * \brief Copy constructors do not make sense for a singleton, disallow copies
     */
    clsparseDeviceTimer( const clsparseDeviceTimer& );

    /**
     * \fn operator=( const StatisticalTimer& )
     * \brief Assignment operator does not make sense for a singleton, disallow assignments
     */
    clsparseDeviceTimer& operator=( const clsparseDeviceTimer& );

    friend std::ostream& operator<<( std::ostream& os, const clsparseDeviceTimer& s );

    //	Calculate the average/mean of data for a given event
    std::vector< StatData > getMean( size_t id );

    //	Calculate the variance of data for a given event
    //	Variance - average of the squared differences between data points and the mean
    std::vector< StatData >	getVariance( size_t id );

    //	Sqrt of variance, also in units of the original data
    std::vector< StatData >	getStdDev( size_t id );

    /**
     * \fn double getAverageTime(size_t id) const
     * \return Return the arithmetic mean of all the samples that have been saved
     */
    std::vector< StatData > getAverageTime( size_t id );

    /**
     * \fn double getMinimumTime(size_t id) const
     * \return Return the arithmetic min of all the samples that have been saved
     */
    std::vector< StatData > getMinimumTime( size_t id );

    void queryOpenCL( size_t id );

    /**
     * \fn void ClearData(void)
     * \brief Clears everything from the class except for labels that have already been
     * created
     */
    void ClearData( );

public:
    /**
     * \fn getInstance()
     * \brief This returns a reference to the singleton timer.  Guarantees only 1 timer class is ever
     *	instantiated within a compilable executable.
     */
    static clsparseDeviceTimer& getInstance( );

    /**
     * \fn void Start( size_t id )
     * \brief Start the timer
     * \sa Stop(), Reset()
     */
    void Start( size_t id );

    /**
     * \fn void Stop( size_t id )
     * \brief Stop the timer
     * \sa Start(), Reset()
     */
    void Stop( size_t id );

    /**
     * \fn void AddSample( const cl_event ev )
     * \brief Explicitely add a timing sample into the class
     */
    virtual void AddSample( std::vector< cl::Event > );

    /**
     * \fn void Clear(void)
     * \brief Clears everything from the class; similar in function to their
     * vector clear method
     */
    void Clear( );

    /**
     * \fn void Reset(void)
     * \brief Prepare the class to recieve new data; this is for reusing the
     * timer instance with new timing data, keeping the same tables and
     * keeping the same sizes
     */
    void Reset( );

    void Reserve( size_t nEvents, size_t nSamples );

    size_t getUniqueID( const std::string& label, cl_uint groupID );

    //	Calculate the average/mean of data for a given event
    void setNormalize( bool norm );

    void Print( cl_ulong flopCount, std::string unit );

    //	Using the stdDev of the entire population (of an id), eliminate those samples that fall
    //	outside some specified multiple of the stdDev.  This assumes that the population
    //	form a gaussian curve.
    size_t	pruneOutliers( cl_double multiple );
    std::vector< size_t > pruneOutliers( size_t id, cl_double multiple );
};

#endif // _STATISTICALTIMER_GPU_H_
