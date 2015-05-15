#include "conjugate_gradients.hpp"

clsparseStatus
clsparseScsrcg(clsparseVector *x,
               const clsparseCsrMatrix *A,
               const clsparseVector *b,
               clSParseSolverControl solverControl, clsparseControl control)
{
    return cg<cl_float>(x, A, b, solverControl, control);
}
