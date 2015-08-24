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
cldenseSaxpy(cldenseVector *r,
             const clsparseScalar *alpha,
             const cldenseVector *x,
             const cldenseVector *y,
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
    clsparse::vector<cl_float> pY (control, y->values, y->num_values);

    assert(pR.size() == pY.size());
    assert(pR.size() == pX.size());

    cl_ulong size = pR.size();

    if(size == 0) return clsparseSuccess;
    //nothing to do
    if (pAlpha[0] == 0.0)
    {
        auto pRBuff = pR.data()();
        auto pYBuff = pY.data()();

        //if R is different pointer than Y than copy Y to R
        if (pRBuff != pYBuff)
        {
            // deep copy;
            pR = pY;
        }
        return clsparseSuccess;
    }

    return axpy(pR, pAlpha, pX, pY, control);
}

clsparseStatus
cldenseDaxpy(cldenseVector *r,
             const clsparseScalar *alpha, const cldenseVector *x,
             const cldenseVector *y,
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
    clsparse::vector<cl_double> pY (control, y->values, y->num_values);

    assert(pR.size() == pY.size());
    assert(pR.size() == pX.size());

    cl_ulong size = pR.size();

    if(size == 0) return clsparseSuccess;

    //nothing to do
    if (pAlpha[0] == 0.0)
    {
        auto pRBuff = pR.data()();
        auto pYBuff = pY.data()();

        //if R is different pointer than Y than copy Y to R
        if (pRBuff != pYBuff)
        {
            // deep copy;
            pR = pY;
        }
        return clsparseSuccess;
    }

    return axpy(pR, pAlpha, pX, pY, control);
}
