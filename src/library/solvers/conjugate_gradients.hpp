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
#include "blas1/cldense_dot.hpp"
#include "blas1/cldense_axpy.hpp"
#include "blas1/cldense_axpby.hpp"
#include "blas1/cldense_nrm1.hpp"

#include "solver_control.hpp"


/*
 * Nice paper describing Conjugate Gradient algorithm can
 * be found here:
 * http://www.cs.cmu.edu/~./quake-papers/painless-conjugate-gradient.pdf
 */
template<typename T, typename PTYPE>
clsparseStatus
cg(clsparseVectorPrivate *pX,
   const clsparseCsrMatrixPrivate* pA,
   const clsparseVectorPrivate *pB,
   PTYPE& M,
   clSParseSolverControl solverControl,
   clsparseControl control)
{

    assert (pA->n == pB->n);
    assert (pA->m == pX->n);
    if ( (pA->n != pB->n) || (pA->m != pX->n) )
    {
        return clsparseInvalidSystemSize;
    }

    //opaque input parameters with clsparse::array type;
    clsparse::vector<T> x(control, pX->values, pX->n);
    clsparse::vector<T> b(control, pB->values, pB->n);

    cl_int status;

    T scalarOne = 1;
    T scalarZero = 0;

    //clsparse::vector<T> norm_b(control, 1, 0, CL_MEM_WRITE_ONLY, true);
    clsparse::scalar<T> norm_b(control, 0, CL_MEM_WRITE_ONLY, false);

    //norm of rhs of equation
    status = Norm1<T>(norm_b, b, control);
    OPENCL_V_THROW(status, "Norm B Failed");

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
    const auto N = pA->n;

    //helper containers, all need to be zeroed
    clsparse::vector<T> y(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> z(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> r(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::vector<T> p(control, N, 0, CL_MEM_READ_WRITE, true);

    clsparse::scalar<T> one(control,  1, CL_MEM_READ_ONLY, true);
    clsparse::scalar<T> zero(control, 0, CL_MEM_READ_ONLY, true);

    // y = A*x
    status = csrmv<T>(one, pA, x, zero, y, control);
    OPENCL_V_THROW(status, "csrmv Failed");

    //r = b - y
    status = r.sub(b, y, control);
    //status = elementwise_transform<T, EW_MINUS>(r, b, y, control);
    OPENCL_V_THROW(status, "b - y Failed");

    clsparse::scalar<T> norm_r(control, 0, CL_MEM_WRITE_ONLY, false);
    status = Norm1<T>(norm_r, r, control);
    OPENCL_V_THROW(status, "norm r Failed");

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
    OPENCL_V_THROW(status, "<r, z> Failed");

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
        OPENCL_V_THROW(status, "csrmv Failed");


        status = dot<T>(yp, y, p, control);
        OPENCL_V_THROW(status, "<y,p> Failed");

        // alpha = <r,z> / <y,p>
        //alpha[0] = rz[0] / yp[0];
        alpha.div(rz, yp, control);

#ifndef NDEBUG
            std::cout << "alpha = " << alpha[0] << std::endl;
#endif

        //x = x + alpha*p
        status = axpy<T>(x, alpha, p, x, control);
        OPENCL_V_THROW(status, "x = x + alpha * p Failed");

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(r, alpha, y, r, control);
        OPENCL_V_THROW(status, "r = r - alpha * y Failed");


        //apply preconditioner z = M*r
        M(r, z, control);

        //store old value of rz
        //improve that by move or swap
        rz_old = rz;

        //rz = <r,z>
        status = dot<T>(rz, r, z, control);
        OPENCL_V_THROW(status, "<r,z> Failed");

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        //beta[0] = rz[0] / rz_old[0];
        beta.div(rz, rz_old, control);
#ifndef NDEBUG
            std::cout << "beta = " << beta[0] << std::endl;
#endif

        //p = z + beta*p;
        status = axpby<T>(p, one, z, beta, p, control );
        OPENCL_V_THROW(status, "p = z + beta*p Failed");

        //calculate norm of r
        status = Norm1<T>(norm_r, r, control);
        OPENCL_V_THROW(status, "norm r Failed");

        //residuum = norm_r[0] / h_norm_b;
        status = residuum.div(norm_r, norm_b, control);
        OPENCL_V_THROW(status, "residuum");

        iteration++;
        converged = solverControl->finished(residuum[0]);

        solverControl->print();
    }
    return clsparseSuccess;
}

#endif //_CLSPARSE_SOLVER_CG_HPP_
