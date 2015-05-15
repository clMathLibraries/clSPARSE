#pragma #once
#ifndef _CLSPARSE_SOLVER_CONTROL_HPP_
#define _CLSPARSE_SOLVER_CONTROL_HPP_

#include "clSPARSE.h"
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "preconditioners/diagonal.hpp"

//enum PRINT_MODE {
//    QUIET =0,
//    NORMAL,
//    VERBOSE
//};

struct _solverControl
{

    _solverControl() : nIters(0), preconditioner(VOID),
        relativeTolerance(0.0), absoluteTolerance(0.0), printMode(NORMAL)
    {

    }

    // number of solver iterations
    cl_int nIters;

    // preconditioner type
    PRECONDITIONER preconditioner;

    // required relative tolerance
    cl_double relativeTolerance;

    // required absolute tolerance
    cl_double absoluteTolerance;

    PRINT_MODE printMode;


};


#endif //_CLSPARSE_SOLVER_CONTROL_HPP_
