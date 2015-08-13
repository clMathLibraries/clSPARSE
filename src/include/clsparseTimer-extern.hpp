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

/* ************************************************************************
* Copyright 2015 Advanced Micro Devices, Inc.
* ************************************************************************/

#pragma once
#ifndef _STATISTICALTIMER_EXTERN_H_
#define _STATISTICALTIMER_EXTERN_H_
#include "clSPARSE.h"
#include "clsparseTimer.hpp"

/**
 * \file clSPARSE.StatisticalTimer.extern.h
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
