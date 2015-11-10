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

/*!
 * \file
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 * This class is implemented entirely in the header, to facilitate inclusion into multiple
 * projects without needing to compile an object file for each project.
 */

#pragma once
#ifndef _STATISTICALTIMER_H_
#define _STATISTICALTIMER_H_

#include <vector>
#include <functional>
#include <string>
#include <stdexcept>

#if defined(__APPLE__) || defined(__MACOSX)
#   include <OpenCL/cl.h>
#else
#   include <CL/cl.h>
#endif

#include "clSPARSE-error.h"

/**
 * \brief Counter that provides a fairly accurate timing mechanism for both
 * windows and linux. This timer is used extensively in all the samples.
 */
class clsparseTimer
{
protected:

    /*!
     * \brief Destructor for StatisticalTimer that cleans up the class
     */
    virtual ~clsparseTimer( )
    {
    };

public:
    /*!
     * \brief Start a timing sample
     */
    virtual void Start( size_t id ) = 0;

    /*!
     * \brief Stop a timing sample
     */
    virtual void Stop( size_t id ) = 0;

    /*!
     * \brief Clear all internal data structures, deallocate all memory
     */
    virtual void Clear( ) = 0;

    /*!
    * \brief Clear all internal data structures, freeing all memory, then reallocate
    * the same amount of memory requested with Reserve
    */
    virtual void Reset( ) = 0;

    /*!
     * \brief The implementation of the timing class may contains data structures
     * that need to grow as more samples are collected.
     * \details This function allows the client to specify to the timing class an
     * estimate for the amount of memory required for the duration of the benchmark
     * \param[in] nEvents  An event is an independent timing location of benchmark interest, such as 
     * two different functions
     * \param[in] nSamples  The number of samples that associated with an event, often a loop-count
     */
    virtual void Reserve( size_t nEvents, size_t nSamples ) = 0;

    /*!
    * \brief Provide a mapping from a 'friendly' human readable text string to an index 
    * into internal data structures
    * \param[in] label  Human readable string to textually identify the event
    * \param[in] groupID  A number paired with the string to differentiate identical strings
    */
    virtual size_t getUniqueID( const std::string& label, cl_uint groupID ) = 0;

    /*!
    * \brief Request that the timer divides total samples collected by sample frequency,
    * converting tick counts into seconds
    * \param[in] norm  True value will divide return result by frequency
    */
    virtual void setNormalize( bool norm ) = 0;

    /*!
    * \brief Print the output of the timing information to the console.  This includes information
    * specific to the derived timer class
    * \param[in] flopCount  The number of flops in a calculation, to calculate the flop rate
    * \param[in] unit  String to represent the Unit to print
    */
    virtual void Print( cl_ulong flopCount, std::string unit ) = 0;

    /*!
    * \brief Using the stdDev of the entire population (of an id), eliminate those samples that fall
    * outside some specified multiple of the stdDev.  This assumes that the population
    * form a Gaussian curve.
    * \param[in] multiple  How many standard deviates to prune outliers, typically integer between 1..3
    */
    virtual size_t	pruneOutliers( cl_double multiple ) = 0;
};

#endif // _STATISTICALTIMER_H_
