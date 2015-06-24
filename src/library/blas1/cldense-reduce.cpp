#include "include/clSPARSE-private.hpp"
#include "internal/kernel-cache.hpp"
#include "internal/kernel-wrap.hpp"

#include "atomic-reduce.hpp"

#include "reduce.hpp"

#include <algorithm>

clsparseStatus
cldenseIreduce(clsparseScalar *s,
               const cldenseVector *x,
               const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_int, RO_PLUS>(pSum, pX, control);
}


clsparseStatus
cldenseSreduce(clsparseScalar *s,
               const cldenseVector *x,
               const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_float, RO_PLUS>(pSum, pX, control);
}

clsparseStatus
cldenseDreduce(clsparseScalar *s,
               const cldenseVector *x,
               const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparseScalarPrivate* pSum = static_cast<clsparseScalarPrivate*> ( s );
    const clsparseVectorPrivate* pX = static_cast<const clsparseVectorPrivate*> ( x );

    return reduce<cl_double, RO_PLUS>(pSum, pX, control);
}
