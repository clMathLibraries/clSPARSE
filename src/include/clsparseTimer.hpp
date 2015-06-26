/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/
#pragma once
#ifndef _STATISTICALTIMER_H_
#define _STATISTICALTIMER_H_

#include <vector>
#include <functional>
#include <string>

#if defined(__APPLE__) || defined(__MACOSX)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include "clSPARSE-error.h"

/**
 * \file clSPARSE.StatisticalTimer.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 *	This class is implemented entirely in the header, to facilitate inclusion into multiple
 *	projects without needing to compile an object file for each project.
 */

//	Definition of a functor object that is passed by reference into the Print statement
//	of the timing class.
//	Functor object to help with accumulating values in vectors
template< typename A, typename R >
class flopsFunc: public std::unary_function < A, R >
{
public:
    virtual typename std::unary_function<A, R>::result_type operator( )( ) = 0;
};

/**
 * \class StatisticalTimer
 * \brief Counter that provides a fairly accurate timing mechanism for both
 * windows and linux. This timer is used extensively in all the samples.
 */
class clsparseTimer
{
protected:
    /**
     * \fn ~clsparseTimer()
     * \brief Destructor for StatisticalTimer that cleans up the class
     */
    virtual ~clsparseTimer( )
    {
    };

    //	friend std::ostream& operator<<( std::ostream& os, const clsparseTimer& s );

public:
    /**
     * \fn void Start( sTimerID id )
     * \brief Start the timer
     * \sa Stop(), Reset()
     */
    virtual void Start( size_t id ) = 0;

    /**
     * \fn void Stop( size_t id )
     * \brief Stop the timer
     * \sa Start(), Reset()
     */
    virtual void Stop( size_t id ) = 0;

    /**
     * \fn void Clear(void)
     * \brief Reset the timer to 0
     * \sa Start(), Stop()
     */
    virtual void Clear( ) = 0;

    /**
     * \fn void Reset(void)
     * \brief Reset the timer to 0
     * \sa Start(), Stop()
     */
    virtual void Reset( ) = 0;

    virtual void Reserve( size_t nEvents, size_t nSamples ) = 0;

    virtual size_t getUniqueID( const std::string& label, cl_uint groupID ) = 0;

    //	Calculate the average/mean of data for a given event
    virtual void setNormalize( bool norm ) = 0;

    virtual void Print( cl_ulong flopCount, std::string unit ) = 0;

    //	Using the stdDev of the entire population (of an id), eliminate those samples that fall
    //	outside some specified multiple of the stdDev.  This assumes that the population
    //	form a gaussian curve.
    virtual size_t	pruneOutliers( cl_double multiple ) = 0;
    //virtual std::vector< size_t > pruneOutliers( size_t id, cl_double multiple ) = 0;
};

#endif // _STATISTICALTIMER_H_
