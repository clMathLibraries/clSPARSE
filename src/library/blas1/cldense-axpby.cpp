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

    return axpby<cl_float>(pR, pAlpha, pX, pBeta, pY, control);
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

    /*TODO: Is it worth to check alpha and beta to call
     * axpy, scale or just copy? It might matter for very large vectors
     *
     * Example of how the code might look like
     *
     * cl_double _alpha = pAlpha[0];
     * cl_double _beta =  pBeta[0];
     *
     * cl_double zero = 0.0;
     *
     * if (_alpha == 0.0)
     * {
     *      if (_beta == 0.0)
     *      {
     *          pR.fill(control, zero);
     *          return clsparseSuccess;
     *      }
     *      else
     *      {
     *          return scale<cl_double>(pR, pBeta, pY);
     *      }
     * }
     * else if ( _beta == 0.0 )
     * {
     *      if ( _alpha == 0.0 )
     *      {
     *          pR.fill(control, zero);
     *          return clsparseSuccess;
     *      }
     *      else
     *      {
     *          return scale<cl_double>(pR, pAlpha, pX);
     *      }
     * }
     */

    return axpby(pR, pAlpha, pX, pBeta, pY, control);
}
