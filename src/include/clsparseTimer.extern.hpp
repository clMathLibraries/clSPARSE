/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/

#pragma once
#ifndef _STATISTICALTIMER_EXTERN_H_
#define _STATISTICALTIMER_EXTERN_H_
#include "clSPARSE.h"
#include "clsparseTimer.hpp"

/**
 * \file clfft.StatisticalTimer.extern.h
 * \brief A timer class that provides a cross platform timer for use
 * in timing code progress with a high degree of accuracy.
 *	This class is implemented entirely in the header, to facilitate inclusion into multiple
 *	projects without needing to compile an object file for each project.
 */
#include "clsparsetimer_export.h"

//	The type of timer to be returned from ::getStatTimer( )
typedef enum clsparseTimerType_
{
    CLSPARSE_GPU = 1,
    CLSPARSE_CPU,
} clsparseTimerType;

//	Table of typedef definitions for all exported functions from this shared module.
//	Clients of this module can use these typedefs to help create function pointers
//	that can be initialized to point to the functions exported from this module.
typedef clsparseTimer* (*PFCLSPARSETIMER)(const clsparseTimerType type);

/**
* \fn getInstance()
* \brief This returns a reference to the singleton timer.  Guarantees only 1 timer class is ever
*	instantiated within a compilable executable.
*/
extern "C" CLSPARSETIMER_EXPORT clsparseTimer* clsparseGetTimer(const clsparseTimerType type);

#endif // _STATISTICALTIMER_EXTERN_H_
