#pragma #once
#ifndef _CLSPARSE_SOLVER_CG_HPP_
#define _CLSPARSE_SOLVER_CG_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"

#include "preconditioners/preconditioner.hpp"
#include "preconditioners/diagonal.hpp"
#include "preconditioners/void.hpp"

//dense vector operations
#include "blas1/reduce.hpp"

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


    assert (pA->n == pB->n);
    assert (pA->m == pX->n);
    if ( (pA->n != pB->n) || (pA->m != pX->n) )
    {
        return clsparseInvalidSystemSize;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;


    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(pA, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(pA, control);
    }


    //start of the solver algorithm
    clsparseScalarPrivate norm_b;
    clsparseInitScalar(&norm_b);

    //let the clMemRAII control the norm_b object
    clMemRAII<T> r_norm_b(control->queue(), &norm_b.value, 1);


    //TODO: Remove PRECISION from reduce template parameters
    // temporary workaround

//    if ()
//        precision = FLOAT;
//    else
//        precision = DOUBLE;


    cl_int status = reduce<T, RO_FABS>(&norm_b, pX, control);








    return clsparseSuccess;

}

#endif //_CLSPARSE_SOLVER_CG_HPP_
