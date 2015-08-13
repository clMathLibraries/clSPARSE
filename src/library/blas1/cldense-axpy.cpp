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
#include "blas1/cldense-axpy.hpp"


clsparseStatus
cldenseSaxpy(cldenseVector *y,
              const clsparseScalar *alpha,
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


    cldenseVectorPrivate* pY = static_cast<cldenseVectorPrivate*> ( y );
    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*> ( alpha );
    const cldenseVectorPrivate* pX = static_cast<const cldenseVectorPrivate*> ( x );

     clMemRAII<cl_float> rAlpha (control->queue(), pAlpha->value);
     cl_float* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->num_values - pY->offset();
     cl_ulong x_size = pX->num_values - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;

    return axpy<cl_float>(size, pY, pAlpha, pX, control);
}

clsparseStatus
cldenseDaxpy(cldenseVector *y,
              const clsparseScalar *alpha, const cldenseVector *x,
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


     clMemRAII<cl_double> rAlpha (control->queue(), pAlpha->value);
     cl_double* fAlpha = rAlpha.clMapMem( CL_TRUE, CL_MAP_READ, pAlpha->offset(), 1);

     //nothing to do
     if (*fAlpha == 0) return clsparseSuccess;

     //leading dimmension is the shorter lenght
     cl_ulong y_size = pY->num_values - pY->offset();
     cl_ulong x_size = pX->num_values - pX->offset();

     cl_ulong size = (x_size >= y_size) ? y_size : x_size;

     if(size == 0) return clsparseSuccess;


    return axpy<cl_double>(size, pY, pAlpha, pX, control);
}
