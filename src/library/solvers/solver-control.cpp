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

#include "clSPARSE.h"
#include "solver-control.hpp"

clsparseCreateSolverResult
clsparseCreateSolverControl(PRECONDITIONER precond, cl_int maxIters,
                            cl_double relTol, cl_double absTol)
{

    clsparseCreateSolverResult cPair;
    cPair.status = clsparseSuccess;
    cPair.control = new _solverControl( );

    if( !cPair.control )
    {
        cPair.control = nullptr;
        cPair.status = clsparseOutOfHostMemory;
        return cPair;
    }

    cPair.control->absoluteTolerance = absTol;
    cPair.control->relativeTolerance = relTol;
    cPair.control->nIters = 0;
    cPair.control->maxIters = maxIters;
    cPair.control->initialResidual = 0;
    cPair.control->preconditioner = precond;

    return cPair;

}

clsparseStatus
clsparseReleaseSolverControl(clSParseSolverControl solverControl)
{

    if (solverControl == nullptr)
    {
        return clsparseInvalidSolverControlObject;
    }

    solverControl->absoluteTolerance = -1;
    solverControl->relativeTolerance = -1;
    solverControl->nIters = -1;
    solverControl->maxIters = -1;
    solverControl->initialResidual = -1;
    solverControl->preconditioner = NOPRECOND;

    delete solverControl;

    solverControl = nullptr;
    return clsparseSuccess;
}

// set the solver control parameters for next use;
clsparseStatus
clsparseSetSolverParams(clSParseSolverControl solverControl,
                        PRECONDITIONER precond,
                        cl_int maxIters, cl_double relTol, cl_double absTol)
{
    if (solverControl == nullptr)
    {
        return clsparseInvalidSolverControlObject;
    }

    solverControl->absoluteTolerance = absTol;
    solverControl->relativeTolerance = relTol;
    solverControl->nIters = 0;
    solverControl->maxIters = maxIters;
    solverControl->initialResidual = 0;
    solverControl->preconditioner = precond;

    return clsparseSuccess;

}

clsparseStatus
clsparseSolverPrintMode(clSParseSolverControl solverControl, PRINT_MODE mode)
{
    if (solverControl == nullptr)
    {
        return clsparseInvalidSolverControlObject;
    }

    solverControl->printMode = mode;

    return clsparseSuccess;
}
