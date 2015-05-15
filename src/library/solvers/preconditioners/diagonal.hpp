#pragma once
#ifndef _CLSPARSE_PREC_DIAGONAL_HPP_
#define _CLSPARSE_PREC_DIAGONAL_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise_transform.hpp"

#include <memory>

template<typename T>
class Diagonal
{
public:
    Diagonal(const clsparseCsrMatrixPrivate* A, clsparseControl control)
    {
        //allocate proper size assuming rectangular size of A;
        cl_uint size = std::min(A->m, A->n);

        clsparseInitVector(&invDiag_A);
        invDiag_A.n = size;

        //r_invDiag_A owns the invDiag so it should release this mem in destructor;
        r_invDiag_A = clMemRAII<T>(control->queue(),
                                  invDiag_A.values, invDiag_A.n,
                                  CL_MEM_READ_WRITE);

        //extract inverse diagonal from A;

    }

    void operator ()(const clsparseVectorPrivate* x,
                     clsparseVectorPrivate* y,
                     clsparseControl control)
    {
        //element wise multiply x*invDiag_A ---> y;
        clsparseStatus status =
                elementwise_transform<T, MULTIPLY>(y, x, &invDiag_A, control);
    }

private:
    //inverse diagonal values of matrix A;
    clsparseVectorPrivate invDiag_A;
    clMemRAII<T> r_invDiag_A;
};

template<typename T>
class PrecondDiagonalHandler : public PreconditionerHandler<T>
{
public:

    using Diag = Diagonal<T>;

    PrecondDiagonalHandler()
    {
    }

    void operator()(const clsparseVectorPrivate* x,
                    clsparseVectorPrivate* y,
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
