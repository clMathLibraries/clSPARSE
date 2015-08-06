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
