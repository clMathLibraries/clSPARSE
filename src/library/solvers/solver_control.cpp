#include "clSPARSE.h"
#if defined(__APPLE__) || defined(__MACOSX)
    #include <OpenCL/cl.hpp>
#else
    #include <CL/cl.hpp>
#endif

#include "solver_control.hpp"

clSParseSolverControl
clsparseCreateSolverControl(cl_int nIters, PRECONDITIONER precond,
                            cl_double relTol, cl_double absTol)
{
    clSParseSolverControl solver_control = new _solverControl();

    if(!solver_control)
    {
        solver_control = nullptr;
    }

    solver_control->absoluteTolerance = absTol;
    solver_control->relativeTolerance = relTol;
    solver_control->nIters = nIters;
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
    solverControl->preconditioner = VOID;

    delete solverControl;

    solverControl = nullptr;
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
