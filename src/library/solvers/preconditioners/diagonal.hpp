#pragma once
#ifndef _CLSPARSE_PREC_DIAGONAL_HPP_
#define _CLSPARSE_PREC_DIAGONAL_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise_transform.hpp"
#include "preconditioner_utils.hpp"
#include "internal/data_types/clarray.hpp"
#include <memory>

/* The simplest preconditioner consists of just the
   inverse values of the diagonal of the matrix:

   The Jacobi preconditioner is one of the simplest forms of preconditioning,
   in which the preconditioner is chosen to be the diagonal of the matrix

                        P = \mathrm{diag}(A).

    Assuming A_{ii} \neq 0, \forall i ,

    we get P^{-1}_{ij} = \frac{\delta_{ij}}{A_{ij}}.

    It is efficient for diagonally dominant matrices A.
*/

template<typename T>
class DiagonalPreconditioner
{
public:
    DiagonalPreconditioner(const clsparseCsrMatrixPrivate* A,
                           clsparseControl control) :
        invDiag_A(control, min(A->m, A->n), 0, CL_MEM_READ_WRITE, false)
    {

        cl_int status;

        // extract inverse diagonal from matrix A and store it in invDiag_A
        // easy to check with poisson matrix;
        status = extract_diagonal<T, true>(invDiag_A, A, control);
        OPENCL_V_THROW(status, "Invalid extract_diagonal kernel execution");

    }

    // apply preconditioner
    void operator ()(const clsparse::array<T>& x,
                     clsparse::array<T>& y,
                     clsparseControl control)
    {
        //element wise multiply y = x*invDiag_A;
        clsparseStatus status =
                elementwise_transform<T, EW_MULTIPLY>(y, x, invDiag_A, control);
        OPENCL_V_THROW(status, "Diagonal operator()");
    }

private:
    //inverse diagonal values of matrix A;
    clsparse::array<T> invDiag_A;
};


template<typename T>
class DiagonalHandler : public PreconditionerHandler<T>
{
public:

    using Diag = DiagonalPreconditioner<T>;

    DiagonalHandler()
    {
    }

    void operator()(const clsparse::array<T>& x,
                    clsparse::array<T>& y,
                    clsparseControl control)
    {
        (*diagonal)(x, y, control);
    }

    void notify(const clsparseCsrMatrixPrivate* pA, clsparseControl control)
    {
        diagonal = std::make_shared<Diag>(pA, control);
    }

private:
    std::shared_ptr<Diag> diagonal;
};

#endif //_CLSPARSE_PREC_DIAGONAL_HPP_
