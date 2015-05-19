#pragma once
#ifndef _CLSPARSE_PREC_VOID_HPP_
#define _CLSPARSE_PREC_VOID_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise_transform.hpp"
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

    void operator() (const clsparseVectorPrivate* x,
                     clsparseVectorPrivate* y,
                     clsparseControl control)
    {
        cl_ulong xSize = x->n - x->offset();
        cl_ulong ySize = y->n - y->offset();

        assert (xSize == ySize);

        //void does nothing just copy x to y;
#if (BUILD_CLVERSION < 200)
        clEnqueueCopyBuffer(control->queue(), x->values, y->values,
                            x->offset(), y->offset(),
                            x->n * sizeof(T),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( )
                            );
#else
        clEnqueueSVMMemcpy(control->queue(), CL_TRUE,
                           y->values, x->values, x->n * sizeof(T),
                           control->event_wait_list.size(),
                           &(control->event_wait_list.front())(),
                           &control->event( ));
#endif
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

    void operator ()(const clsparseVectorPrivate* x,
                     clsparseVectorPrivate* y,
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
