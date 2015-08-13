/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
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

    _solverControl() : nIters(0), maxIters(0), preconditioner(NOPRECOND),
        relativeTolerance(0.0), absoluteTolerance(0.0),
        initialResidual(0), currentResidual(0), printMode(QUIET)
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

    void print()
    {
        if (printMode == VERBOSE)
        {
            std::cout << "Iteration: " << nIters
                      << " Residuum: " << currentResidual
                      << std::endl;
        }
    }

    std::string printPreconditioner()
    {

        switch(preconditioner)
        {
        case NOPRECOND:
            return "No preconditioner";
        case DIAGONAL:
            return "Diagonal";
        }
    }

    void printSummary(clsparseStatus status)
    {
        if (printMode == VERBOSE || printMode == NORMAL)
        {
            std::cout << "Solver constraints: \n"
                      << "\trelative tolerance = " << relativeTolerance
                      << "\n\tabsolute tolerance = " << absoluteTolerance
                      << "\n\tmax iterations = " << maxIters
                      << "\n\tPreconditioner: " << printPreconditioner()
                      << std::endl;

            std::cout << "Solver finished calculations with status "
                      << status << std::endl;

            std::cout << "\tfinal residual = " << currentResidual
                      << "\titerations = " << nIters
                      << std::endl;
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
