#include "conjugate_gradients.hpp"


clsparseStatus
clsparseScsrcg(clsparseVector *x, clsparseCsrMatrix *A, clsparseVector *b,
               clSParseSolverControl solverControl, clsparseControl control)
{
    return cg<cl_float>(x, A, b, solverControl, control);
}
