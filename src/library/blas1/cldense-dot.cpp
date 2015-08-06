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
#include "cldense-dot.hpp"
clsparseStatus
cldenseSdot (clsparseScalar* r,
             const cldenseVector* x,
             const cldenseVector* y,
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


    clsparseScalarPrivate* pR = static_cast<clsparseScalarPrivate*>( r );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate* pY = static_cast<const cldenseVectorPrivate*> ( y );

    return dot<cl_float>(pR, pX, pY, control);
}

clsparseStatus
cldenseDdot (clsparseScalar* r,
             const cldenseVector* x,
             const cldenseVector* y,
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


    clsparseScalarPrivate* pDot = static_cast<clsparseScalarPrivate*>( r );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate* pY = static_cast<const cldenseVectorPrivate*> ( y );

    return dot<cl_double>(pDot, pX, pY, control);
}
