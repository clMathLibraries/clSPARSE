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

#pragma once
#ifndef _CLSPARSE_SOLVER_CG_HPP_
#define _CLSPARSE_SOLVER_CG_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-internal.hpp"
#include "internal/data-types/clvector.hpp"
#include "internal/data-types/clarray.hpp"

#include "preconditioners/preconditioner.hpp"
#include "preconditioners/diagonal.hpp"
#include "preconditioners/void.hpp"

//matrix multiply
#include "blas2/clsparse-csrmv.hpp"

//dense vector operations
#include "blas1/cldense-dot.hpp"
#include "blas1/cldense-axpy.hpp"
#include "blas1/cldense-axpby.hpp"
#include "blas1/cldense-nrm1.hpp"

#include "solver-control.hpp"


/*
 * Nice paper describing Conjugate Gradient algorithm can
 * be found here:
 * http://www.cs.cmu.edu/~./quake-papers/painless-conjugate-gradient.pdf
 */
template<typename T, typename PTYPE>
clsparseStatus
cg(cldenseVectorPrivate *pX,
   const clsparseCsrMatrixPrivate* pA,
   const cldenseVectorPrivate *pB,
   PTYPE& M,
   clSParseSolverControl solverControl,
   clsparseControl control)
{

    assert( pA->num_rows == pB->num_values );
    assert( pA->num_cols == pX->num_values );
    if( ( pA->num_rows != pB->num_values ) || ( pA->num_cols != pX->num_values ) )
    {
        return clsparseInvalidSystemSize;
    }

    //opaque input parameters with clsparse::array type;
    clsparse::vector<T> x(control, pX->values, pX->num_values);
    clsparse::vector<T> b(control, pB->values, pB->num_values);

    cl_int status;

    T scalarOne = 1;
    T scalarZero = 0;

    //clsparse::vector<T> norm_b(control, 1, 0, CL_MEM_WRITE_ONLY, true);
    clsparse::scalar<T> norm_b(control, 0, CL_MEM_WRITE_ONLY, false);

    //norm of rhs of equation
    status = Norm1<T>(norm_b, b, control);
    CLSPARSE_V(status, "Norm B Failed");

    //norm_b is calculated once
    T h_norm_b = norm_b[0];

#ifndef NDEBUG
    std::cout << "norm_b " << h_norm_b << std::endl;
#endif

    if (h_norm_b == 0) //special case b is zero so solution is x = 0
    {
        solverControl->nIters = 0;
        solverControl->absoluteTolerance = 0.0;
        solverControl->relativeTolerance = 0.0;
        //we can either fill the x with zeros or cpy b to x;
        x = b;
        return clsparseSuccess;
    }


    //continuing "normal" execution of cg algorithm
    const auto N = pA->num_rows;

    //helper containers, all need to be zeroed
    clsparse::vector<T> y(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> z(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> r(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> p(control, N, 0, CL_MEM_READ_WRITE, true);

    clsparse::scalar<T> one(control,  1, CL_MEM_READ_ONLY, true);
    clsparse::scalar<T> zero(control, 0, CL_MEM_READ_ONLY, true);

    // y = A*x
    status = csrmv<T>(one, pA, x, zero, y, control);
    CLSPARSE_V(status, "csrmv Failed");

    //r = b - y
    status = r.sub(b, y, control);
    //status = elementwise_transform<T, EW_MINUS>(r, b, y, control);
    CLSPARSE_V(status, "b - y Failed");

    clsparse::scalar<T> norm_r(control, 0, CL_MEM_WRITE_ONLY, false);
    status = Norm1<T>(norm_r, r, control);
    CLSPARSE_V(status, "norm r Failed");

    //T residuum = 0;
    clsparse::scalar<T> residuum(control, 0, CL_MEM_WRITE_ONLY, false);

    //residuum = norm_r[0] / h_norm_b;
    residuum.div(norm_r, norm_b, control);

    solverControl->initialResidual = residuum[0];
#ifndef NDEBUG
        std::cout << "initial residuum = "
                  << solverControl->initialResidual << std::endl;
#endif
    if (solverControl->finished(solverControl->initialResidual))
    {
        solverControl->nIters = 0;
        return clsparseSuccess;
    }
    //apply preconditioner z = M*r
    M(r, z, control);

    //copy inital z to p
    p = z;

    //rz = <r, z>, here actually should be conjugate(r)) but we do not support complex type.
    clsparse::scalar<T> rz(control, 0, CL_MEM_WRITE_ONLY, false);
    status = dot<T>(rz, r, z, control);
    CLSPARSE_V(status, "<r, z> Failed");

    int iteration = 0;

    bool converged = false;

    clsparse::scalar<T> alpha (control, 0, CL_MEM_READ_WRITE, false);
    clsparse::scalar<T> beta  (control, 0, CL_MEM_READ_WRITE, false);

    //yp buffer for inner product of y and p vectors;
    clsparse::scalar<T> yp(control, 0, CL_MEM_WRITE_ONLY, false);

    clsparse::scalar<T> rz_old(control, 0, CL_MEM_WRITE_ONLY, false);

    while(!converged)
    {
        solverControl->nIters = iteration;

        //y = A*p
        status = csrmv<T>(one, pA, p, zero, y, control);
        CLSPARSE_V(status, "csrmv Failed");


        status = dot<T>(yp, y, p, control);
        CLSPARSE_V(status, "<y,p> Failed");

        // alpha = <r,z> / <y,p>
        //alpha[0] = rz[0] / yp[0];
        alpha.div(rz, yp, control);

#ifndef NDEBUG
            std::cout << "alpha = " << alpha[0] << std::endl;
#endif

        //x = x + alpha*p
        status = axpy<T>(x, alpha, p, x, control);
        CLSPARSE_V(status, "x = x + alpha * p Failed");

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(r, alpha, y, r, control);
        CLSPARSE_V(status, "r = r - alpha * y Failed");


        //apply preconditioner z = M*r
        M(r, z, control);

        //store old value of rz
        //improve that by move or swap
        rz_old = rz;

        //rz = <r,z>
        status = dot<T>(rz, r, z, control);
        CLSPARSE_V(status, "<r,z> Failed");

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        //beta[0] = rz[0] / rz_old[0];
        beta.div(rz, rz_old, control);
#ifndef NDEBUG
            std::cout << "beta = " << beta[0] << std::endl;
#endif

        //p = z + beta*p;
        status = axpby<T>(p, one, z, beta, p, control );
        CLSPARSE_V(status, "p = z + beta*p Failed");

        //calculate norm of r
        status = Norm1<T>(norm_r, r, control);
        CLSPARSE_V(status, "norm r Failed");

        //residuum = norm_r[0] / h_norm_b;
        status = residuum.div(norm_r, norm_b, control);
        CLSPARSE_V(status, "residuum");

        iteration++;
        converged = solverControl->finished(residuum[0]);

        solverControl->print();
    }
    return clsparseSuccess;
}

#endif //_CLSPARSE_SOLVER_CG_HPP_
