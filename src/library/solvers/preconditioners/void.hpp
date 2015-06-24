#pragma once
#ifndef _CLSPARSE_PREC_VOID_HPP_
#define _CLSPARSE_PREC_VOID_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise-transform.hpp"
#include "preconditioner_utils.hpp"
#include <memory>


/*
 * Void preconditioner, does nothing it is just to fit the solver structure.
 * This is the default value of the solver control.
 */
template <typename T>
class VoidPreconditioner
{
public:

    VoidPreconditioner (const clsparseCsrMatrixPrivate* A, clsparseControl control)
    {

    }

    void operator() (const clsparse::vector<T>& x,
                     clsparse::vector<T>& y,
                     clsparseControl control)
    {

        assert (x.size() == y.size());

        //void does nothing just copy x to y;

        //deep copy;
        y = x;
    }

};

template <typename T>
class VoidHandler : public PreconditionerHandler<T>
{
public:

    using Void = VoidPreconditioner<T>;

    VoidHandler()
    {

    }

    void operator ()(const clsparse::vector<T>& x,
                     clsparse::vector<T>& y,
                     clsparseControl control)
    {
        (*void_precond)(x, y, control);
    }

    void notify(const clsparseCsrMatrixPrivate *pA,
                clsparseControl control)
    {
        void_precond = std::make_shared<Void> (pA, control);
    }

private:
    std::shared_ptr<Void> void_precond;
};

#endif //_CLSPARSE_PREC_VOID_HPP_
