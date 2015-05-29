#pragma once
#ifndef _CL_SPARSE_xx_H_
#define _CL_SPARSE_xx_H_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

//  Used to help define the orientation of a dense matrix
typedef enum _cldenseMajor
{
    rowMajor = 0,
    columnMajor
} cldenseMajor;

#endif
