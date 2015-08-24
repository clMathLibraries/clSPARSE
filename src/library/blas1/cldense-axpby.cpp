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
#include "cldense-axpy.hpp"
#include "cldense-scale.hpp"

namespace internal
{

// this will trigger the right proper code depends on the
// alpha and beta parameters;
template<typename T>
clsparseStatus
axpby(clsparse::vector<T>& pR,
     const clsparse::vector<T>& pAlpha,
     const clsparse::vector<T>& pX,
     const clsparse::vector<T>& pBeta,
     const clsparse::vector<T>& pY,
     const clsparseControl control)
{
    //check if we can run ligher code;
    T _alpha = pAlpha[0];
    T _beta =  pBeta[0];

    T zero = 0.0;

    if (_alpha == 0.0)
    {
        if (_beta == 0.0)
        {
            pR.fill(control, zero);
            return clsparseSuccess;
        }
        else
        {
            // r = beta * y
            return scale(pR, pBeta, pY, control);
        }
    }

    else if ( _beta == 0.0 )
    {
        if ( _alpha == 0.0 )
        {
            pR.fill(control, zero);
            return clsparseSuccess;
        }
        else
        {
            // r = alpha * x
            return scale(pR, pAlpha, pX, control);
        }
    }

    // r = by + x;
    else if ( _alpha == 1.0 )
    {
        return axpy(pR, pBeta, pY, pX, control);
    }

    // r = ax + y;
    else if ( _beta == 1.0 )
    {
        return axpy(pR, pAlpha, pX, pY, control);
    }

    //reach out of internal namespace;
    return ::axpby(pR, pAlpha, pX, pBeta, pY, control);
}

} // namespace internal


clsparseStatus
cldenseSaxpby(cldenseVector *r,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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


    clsparse::vector<cl_float> pR (control, r->values, r->num_values);
    clsparse::vector<cl_float> pAlpha(control, alpha->value, 1);
    clsparse::vector<cl_float> pX (control, x->values, x->num_values);
    clsparse::vector<cl_float> pBeta(control, beta->value, 1);
    clsparse::vector<cl_float> pY (control, y->values, y->num_values);

    cl_ulong size = pR.size();

    if(size == 0) return clsparseSuccess;

    return internal::axpby(pR, pAlpha, pX, pBeta, pY, control);
}

clsparseStatus
cldenseDaxpby(cldenseVector *r,
               const clsparseScalar *alpha,
               const cldenseVector *x,
               const clsparseScalar *beta,
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

    clsparse::vector<cl_double> pR (control, r->values, r->num_values);
    clsparse::vector<cl_double> pAlpha(control, alpha->value, 1);
    clsparse::vector<cl_double> pX (control, x->values, x->num_values);
    clsparse::vector<cl_double> pBeta(control, beta->value, 1);
    clsparse::vector<cl_double> pY (control, y->values, y->num_values);

    cl_ulong size = pR.size();

    if(size == 0) return clsparseSuccess;

    return internal::axpby(pR, pAlpha, pX, pBeta, pY, control);
}
