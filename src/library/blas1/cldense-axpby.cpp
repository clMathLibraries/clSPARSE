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
#include "cldense-axpby.hpp"

clsparseStatus
cldenseSaxpby(cldenseVector *y,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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


    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );

    //is it necessary? Maybe run the kernel nevertheless those values?
//    clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
//    cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

//    clMemRAII<cl_float> rBeta (control->queue(), pBeta->value);
//    cl_float* fBeta = rBeta.clMapMem( CL_TRUE, CL_MAP_READ, pBeta->offset(), 1);

    //nothing to do
    //if (*fAlpha == 0) return clsparseSuccess;

    cl_ulong y_size = pY->num_values - pY->offset();
    cl_ulong x_size = pX->num_values - pX->offset();

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;



    return axpby<cl_float>(size, pY, pAlpha, pX, pBeta, control);
}

clsparseStatus
cldenseDaxpby(cldenseVector *y,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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

    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*> ( beta );


    cl_ulong y_size = pY->num_values - pY->offset();
    cl_ulong x_size = pX->num_values - pX->offset();

    cl_ulong size = (x_size >= y_size) ? y_size : x_size;

    if(size == 0) return clsparseSuccess;


    return axpby<cl_double>(size, pY, pAlpha, pX, pBeta, control);
}
