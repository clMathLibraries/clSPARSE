#pragma once
#ifndef _CLSPARSE_PRECONDITIONER_HPP_
#define _CLSPARSE_PRECONDITIONER_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/data_types/clarray.hpp"

//enum PRECONDITIONER
//{
//    VOID = 0,
//    DIAGONAL
//};

/** //From Wikipedia
In linear algebra and numerical analysis, a preconditioner P of a matrix A is
a matrix such that P^{-1}A has a smaller condition number than A.
It is also common to call T=P^{-1} the preconditioner, rather than P, since P
itself is rarely explicitly available. In modern preconditioning, the application
of T=P^{-1}, i.e., multiplication of a column vector, or a block of column vectors,
by T=P^{-1}, is commonly performed by rather sophisticated computer software
packages in a matrix-free fashion, i.e., where neither P, nor T=P^{-1}
(and often not even A) are explicitly available in a matrix form.

Preconditioners are useful in iterative methods to solve a linear system Ax=b
for x since the rate of convergence for most iterative linear solvers increases
as the condition number of a matrix decreases as a result of preconditioning.
Preconditioned iterative solvers typically outperform direct solvers,
e.g., Gaussian elimination, for large, especially for sparse, matrices.
Iterative solvers can be used as matrix-free methods, i.e. become the only choice
if the coefficient matrix A is not stored explicitly, but is accessed by
evaluating matrix-vector products.

Description

Instead of solving the original linear system Ax=b, one may solve either the
right preconditioned system:

    AP^{-1}Px = b

via solving

    AP^{-1}y=b

for y and

    Px=y

for x; or the left preconditioned system:

    P^{-1}(Ax-b)=0

both of which give the same solution as the original system so long as the
preconditioner matrix P is nonsingular. The left preconditioning is more common.
The goal of this preconditioned system is to reduce the condition number of the
left or right preconditioned system matrix P^{-1}A or AP^{-1}, respectively.
The preconditioned matrix P^{-1}A or AP^{-1} is almost never explicitly formed.
Only the action of applying the preconditioner solve operation P^{-1} to a given
vector need to be computed in iterative methods.
*/

/* General structure of preconditioner handler used in solver.
   Every other preconditioner should comply this structure, as a proof of concept
   it is enough but for AMG we will require something more complex
*/

template<typename T>
class PreconditionerHandler
{
public:
    virtual void operator()(const clsparse::array<T>& x,
                       clsparse::array<T>& y,
                       clsparseControl control) = 0;

    virtual void notify(const clsparseCsrMatrixPrivate* pA,
                        clsparseControl control) = 0;
};

#endif //_CLSPARSE_PRECONDITIONER_HPP_
