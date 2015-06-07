#include "clSPARSE.h"
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "solver_control.hpp"

clSParseSolverControl
clsparseCreateSolverControl(PRECONDITIONER precond, cl_int maxIters,
                            cl_double relTol, cl_double absTol)
{
    clSParseSolverControl solver_control = new _solverControl();

    if(!solver_control)
    {
        solver_control = nullptr;
    }

    solver_control->absoluteTolerance = absTol;
    solver_control->relativeTolerance = relTol;
    solver_control->nIters = 0;
    solver_control->maxIters = maxIters;
    solver_control->initialResidual = 0;
    solver_control->preconditioner = precond;

    return solver_control;

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
