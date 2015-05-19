#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include "atomic_reduce.hpp"

#include "reduce.hpp"

#include <algorithm>


clsparseStatus
cldenseSreduce(clsparseScalar *s,
               const clsparseVector *x,
               const clsparseControl control)
{
    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_float, RO_PLUS>(pSum, pX, control);
}

clsparseStatus
cldenseDreduce(clsparseScalar *s,
               const clsparseVector *x,
               const clsparseControl control)
{
    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_double, RO_PLUS>(pSum, pX, control);
}
