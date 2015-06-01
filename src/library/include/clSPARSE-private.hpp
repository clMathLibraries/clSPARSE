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
#include "include/clSPARSE_1x.hpp"
#else
#include "include/clSPARSE_2x.hpp"
#endif

// Constants used to help generate kernels for the CSR adaptive algorithm; used between coo2csr and csrmv_adaptive
const cl_uint WG_BITS = 24;
const cl_uint ROW_BITS = 32;
const cl_uint BLKSIZE = 1024;

#endif