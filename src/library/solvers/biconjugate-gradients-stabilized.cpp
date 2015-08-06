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

#include "biconjugate-gradients-stabilized.hpp"

clsparseStatus
clsparseScsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control)
{
    using T = cl_float;

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

    cldenseVectorPrivate* pX = static_cast<cldenseVectorPrivate*> ( x );
    const clsparseCsrMatrixPrivate* pA = static_cast<const clsparseCsrMatrixPrivate*> ( A );
    const cldenseVectorPrivate* pB = static_cast<const cldenseVectorPrivate*> ( b );


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

    clsparseStatus status = bicgStab<T>(pX, pA, pB, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;


}

clsparseStatus
clsparseDcsrbicgStab(cldenseVector* x, const clsparseCsrMatrix *A, const cldenseVector *b,
               clSParseSolverControl solverControl, clsparseControl control)
{
    using T = cl_double;

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

    cldenseVectorPrivate* pX = static_cast<cldenseVectorPrivate*> ( x );
    const clsparseCsrMatrixPrivate* pA = static_cast<const clsparseCsrMatrixPrivate*> ( A );
    const cldenseVectorPrivate* pB = static_cast<const cldenseVectorPrivate*> ( b );


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

    clsparseStatus status = bicgStab<T>(pX, pA, pB, *preconditioner, solverControl, control);

    solverControl->printSummary(status);

    return status;
}
