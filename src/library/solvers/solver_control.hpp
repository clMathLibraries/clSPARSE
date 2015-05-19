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

    _solverControl() : nIters(0), maxIters(0), preconditioner(VOID),
        relativeTolerance(0.0), absoluteTolerance(0.0),
        initialResidual(0), currentResidual(0), printMode(NORMAL)
    {

    }

    bool finished(const cl_double residuum)
    {
        return converged(residuum) || nIters >= maxIters;
    }

    bool converged(const cl_double residuum)
    {
        currentResidual = residuum;
        if(residuum <= relativeTolerance ||
           residuum <= absoluteTolerance * initialResidual)
        {
            return true;
        }
        else
        {
            return false;
        }
    }

    // current solver iteration;
    cl_int nIters;

    // maximum solver iterations;
    cl_int maxIters;

    // preconditioner type
    PRECONDITIONER preconditioner;

    // required relative tolerance
    cl_double relativeTolerance;

    // required absolute tolerance
    cl_double absoluteTolerance;

    cl_double initialResidual;

    cl_double currentResidual;

    PRINT_MODE printMode;
};


#endif //_CLSPARSE_SOLVER_CONTROL_HPP_
