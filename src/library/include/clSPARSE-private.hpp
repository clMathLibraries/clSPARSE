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
#ifndef _CL_SPARSE_PRIVATE_HPP_
#define _CL_SPARSE_PRIVATE_HPP_
// Definitions and #includes private to the internal implementation of the library

#if defined ( _WIN32 )
#define NOMINMAX
#endif

#include "clSPARSE.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// Include appropriate data type definitions appropriate to the cl version supported
#if( BUILD_CLVERSION < 200 )
#include "include/clSPARSE-1x.hpp"
#else
#include "include/clSPARSE-2x.hpp"
#endif

// Constants used to help generate kernels for the CSR adaptive algorithm; used between coo2csr and csrmv_adaptive
const cl_uint WG_BITS = 24;
const cl_uint ROW_BITS = 32;
const cl_uint BLKSIZE = 1024;
const cl_uint BLOCK_MULTIPLIER = 3;
const cl_uint ROWS_FOR_VECTOR = 1;

#endif
