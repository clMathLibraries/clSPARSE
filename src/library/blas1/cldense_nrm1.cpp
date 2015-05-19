#include "include/clSPARSE-private.hpp"
#include "internal/kernel_cache.hpp"
#include "internal/kernel_wrap.hpp"

#include "reduce.hpp"

#include <algorithm>

clsparseStatus
cldenseSnrm1(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_float, RO_FABS>(pS, pX, control);

}

clsparseStatus
cldenseDnrm1(clsparseScalar* s,
             const clsparseVector* x,
             const clsparseControl control)
{
    clsparseScalarPrivate* pS = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_double, RO_FABS>(pS, pX, control);
}
