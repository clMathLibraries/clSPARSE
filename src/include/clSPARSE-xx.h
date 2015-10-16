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

/*! \file
* \brief clSPARSE-xx.h defines public types used by any OpenCL version
*/

#pragma once
#ifndef _CL_SPARSE_xx_H_
#define _CL_SPARSE_xx_H_

#if defined( __APPLE__ ) || defined( __MACOSX )
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

/*! \brief For dense data structures, this enum specifies how multi-dimensional data structures
 * are laid out in memory.  rowMajor corresponds to the 'C' language storage order, and
 * columnMajor corresponds to the 'Fortran' language storage order
*/
typedef enum _cldenseMajor
{
    rowMajor = 1,
    columnMajor
} cldenseMajor;

#endif
