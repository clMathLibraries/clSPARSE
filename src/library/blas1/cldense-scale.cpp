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
#include "cldense-scale.hpp"

clsparseStatus
cldenseSscale ( cldenseVector* r,
                const clsparseScalar* alpha,
                const cldenseVector* y,
                const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparse::vector<cl_float> pR(control, r->values, r->num_values );
    clsparse::vector<cl_float> pAlpha(control, alpha->value, 1);
    clsparse::vector<cl_float> pY(control, y->values, y->num_values );

    assert(r->num_values == y->num_values);

    cl_float pattern = 0.0f;

    if (pAlpha[0] == 0.f)
    {

        cl_int status = pR.fill(control, pattern);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    return scale(pR, pAlpha, pY, control);
}


clsparseStatus
cldenseDscale( cldenseVector* r,
               const clsparseScalar* alpha,
               const cldenseVector* y,
               const clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    clsparse::vector<cl_double> pR (control, r->values, r->num_values );
    clsparse::vector<cl_double> pAlpha (control, alpha->value, 1);
    clsparse::vector<cl_double> pY (control, y->values, y->num_values );

    assert(r->num_values == y->num_values);

    cl_double pattern = 0.0;

    if (pAlpha[0] == 0.0)
    {

        cl_int status = pR.fill(control, pattern);

        if (status != CL_SUCCESS)
        {
            return clsparseInvalidKernelExecution;
        }

        return clsparseSuccess;
    }

    return scale(pR, pAlpha, pY, control);
}
