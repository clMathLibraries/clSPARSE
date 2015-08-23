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
#include "elementwise-transform.hpp"

// +
clsparseStatus
cldenseSadd( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_float, EW_PLUS> (pR, pX, pY, control);
}

clsparseStatus
cldenseDadd( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_double, EW_PLUS> (pR, pX, pY, control);
}

// -
clsparseStatus
cldenseSsub( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_float, EW_MINUS> (pR, pX, pY, control);
}

clsparseStatus
cldenseDsub( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );
    return elementwise_transform<cl_double, EW_MINUS> (pR, pX, pY, control);
}

// *
clsparseStatus
cldenseSmul( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_float, EW_MULTIPLY> (pR, pX, pY, control);
}

clsparseStatus
cldenseDmul( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_double, EW_MULTIPLY> (pR, pX, pY, control);
}

// "/" (div)
clsparseStatus
cldenseSdiv( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_float, EW_DIV> (pR, pX, pY, control);
}

clsparseStatus
cldenseDdiv( cldenseVector* r,
             const cldenseVector* x,
             const cldenseVector* y,
             const clsparseControl control )
{
    cldenseVectorPrivate *pR = static_cast<cldenseVectorPrivate*> ( r );
    const cldenseVectorPrivate *pX = static_cast<const cldenseVectorPrivate*> ( x );
    const cldenseVectorPrivate *pY = static_cast<const cldenseVectorPrivate*> ( y );

    return elementwise_transform<cl_double, EW_DIV> (pR, pX, pY, control);
}
