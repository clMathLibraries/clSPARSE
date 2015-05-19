#pragma #once
#ifndef _CLSPARSE_SOLVER_CG_HPP_
#define _CLSPARSE_SOLVER_CG_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"

#include "preconditioners/preconditioner.hpp"
#include "preconditioners/diagonal.hpp"
#include "preconditioners/void.hpp"

#include "solver_control.hpp"

template<typename T>
clsparseStatus
cg(clsparseVector *x,
   const clsparseCsrMatrix* A,
   const clsparseVector *b,
   clSParseSolverControl solverControl,
   clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    if (solverControl == nullptr)
    {
        return clsparseInvalidSolverControlObject;
    }

    clsparseVectorPrivate* pX = static_cast<clsparseVectorPrivate*> ( x );
    const clsparseCsrMatrixPrivate* pA = static_cast<const clsparseCsrMatrixPrivate*> ( A );
    const clsparseVectorPrivate* pB = static_cast<const clsparseVectorPrivate*> ( b );

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;


    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of Diag class
        preconditioner->notify(pA, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());

        preconditioner->notify(pA, control);
    }

    preconditioner->operator ()(pX, pX, control);


    return clsparseSuccess;

}

#endif //_CLSPARSE_SOLVER_CG_HPP_
