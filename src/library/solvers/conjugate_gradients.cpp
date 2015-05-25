#include "conjugate_gradients.hpp"
clsparseStatus
clsparseScsrcg(clsparseVector *x,
               const clsparseCsrMatrix *A,
               const clsparseVector *b,
               clSParseSolverControl solverControl,
               clsparseControl control)
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

    clsparseVectorPrivate* pX = static_cast<clsparseVectorPrivate*> ( x );
    const clsparseCsrMatrixPrivate* pA = static_cast<const clsparseCsrMatrixPrivate*> ( A );
    const clsparseVectorPrivate* pB = static_cast<const clsparseVectorPrivate*> ( b );


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

    clsparseStatus status = cg<cl_float>(pX, pA, pB, *preconditioner, solverControl, control);
    std::cout << "nIters = " << solverControl->nIters << std::endl;
    return status;
}
