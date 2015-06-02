#pragma once
#ifndef _CLSPARSE_SOLVER_CG_HPP_
#define _CLSPARSE_SOLVER_CG_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"
#include "internal/data_types/clarray.hpp"

#include "preconditioners/preconditioner.hpp"
#include "preconditioners/diagonal.hpp"
#include "preconditioners/void.hpp"

//matrix multiply
#include "spmv/clsparse_csrmv.hpp"
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
    clsparse::array<T> x(control, pX->values, pX->n);
    clsparse::array<T> b(control, pB->values, pB->n);

    cl_int status;

    T scalarOne = 1;
    T scalarZero = 0;

    clsparse::array<T> norm_b(control, 1, 0, CL_MEM_WRITE_ONLY, true);

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
    clsparse::array<T> y(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::array<T> z(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::array<T> r(control, N, 0, CL_MEM_READ_WRITE, true);
    clsparse::array<T> p(control, N, 0, CL_MEM_READ_WRITE, true);

    clsparse::array<T> one(control, 1, 1, CL_MEM_READ_ONLY, true);
    clsparse::array<T> zero(control, 1, 0, CL_MEM_READ_ONLY, true);

    // y = A*x
    status = csrmv<T>(one, pA, x, zero, y, control);
    OPENCL_V_THROW(status, "csrmv Failed");

    //r = b - y
    status = elementwise_transform<T, EW_MINUS>(r, b, y, control);
    OPENCL_V_THROW(status, "b - y Failed");

    clsparse::array<T> norm_r(control, 1, 0, CL_MEM_WRITE_ONLY, true);
    status = Norm1<T>(norm_r, r, control);
    OPENCL_V_THROW(status, "norm r Failed");

    T residuum = 0;
    {

        residuum = norm_r[0] / h_norm_b;
#ifndef NDEBUG
        std::cout << "initial residuum = " << residuum << std::endl;
#endif
    }

    solverControl->initialResidual = residuum;
    if (solverControl->finished(residuum))
    {
        solverControl->nIters = 0;
        return clsparseSuccess;
    }
    //apply preconditioner z = M*r
    M(r, z, control);

    //copy inital z to p
    p = z;

    //rz = <r, z>, here actually should be conjugate(r)) but we do not support complex type.
    clsparse::array<T> rz(control, 1, 0, CL_MEM_WRITE_ONLY, true);
    status = dot<T>(rz, r, z, control);
    OPENCL_V_THROW(status, "<r, z> Failed");

    int iteration = 0;

    bool converged = false;

    clsparse::array<T> alpha (control, 1, 0, CL_MEM_READ_WRITE, true);
    clsparse::array<T> beta  (control, 1, 0, CL_MEM_READ_WRITE, true);

    //yp buffer for inner product of y and p vectors;
    clsparse::array<T> yp(control, 1, 0, CL_MEM_WRITE_ONLY, true);

    clsparse::array<T> rz_old(control, 1, 0, CL_MEM_WRITE_ONLY, true);

    while(!converged)
    {
        solverControl->nIters = iteration;

        //y = A*p
        status = csrmv<T>(one, pA, p, zero, y, control);
        OPENCL_V_THROW(status, "csrmv Failed");


        status = dot<T>(yp, y, p, control);
        OPENCL_V_THROW(status, "<y,p> Failed");

        // alpha = <r,z> / <y,p>
        alpha[0] = rz[0] / yp[0];

#ifndef NDEBUG
            std::cout << "alpha = " << alpha[0] << std::endl;
#endif

        //x = x + alpha*p
        status = axpy<T>(x, alpha, p, control);
        OPENCL_V_THROW(status, "x = x + alpha * p Failed");

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(r, alpha, y, control);
        OPENCL_V_THROW(status, "r = r - alpha * y Failed");


        //apply preconditioner z = M*r
        M(r, z, control);

        //store old value of rz
        rz_old = rz;

        //rz = <r,z>
        status = dot<T>(rz, r, z, control);
        OPENCL_V_THROW(status, "<r,z> Failed");

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        beta[0] = rz[0] / rz_old[0];
#ifndef NDEBUG
            std::cout << "beta = " << beta[0] << std::endl;
#endif

        //p = z + beta*p;
        status = axpby<T>(p, one, z, beta, control );
        OPENCL_V_THROW(status, "p = z + beta*p Failed");

        //calculate norm of r
        status = Norm1<T>(norm_r, r, control);
        OPENCL_V_THROW(status, "norm r Failed");

        residuum = norm_r[0] / h_norm_b;

        iteration++;
        converged = solverControl->finished(residuum);

        solverControl->print();
    }

    return clsparseSuccess;
}

#endif //_CLSPARSE_SOLVER_CG_HPP_
