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
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );

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
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );

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
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );

    return reduce<cl_double, RO_PLUS>(pSum, pX, control);
}
