#pragma once
#ifndef _CLSPARSE_SOLVER_BICGSTAB_HPP_
#define _CLSPARSE_SOLVER_BICGSTAB_HPP_

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

template <typename T, typename PTYPE>
clsparseStatus
bicgStab(clsparseVectorPrivate *pX,
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

    typedef typename clsparse::vector<T> vector;
    typedef typename clsparse::scalar<T> scalar;

    //opaque input parameters with clsparse::array type;
    vector x(control, pX->values, pX->n);
    vector b(control, pB->values, pB->n);

    cl_int status;

    scalar norm_b(control, 0, CL_MEM_WRITE_ONLY, false);

    //norm of rhs of equation
    status = Norm1<T>(norm_b, b, control);
    OPENCL_V_THROW(status, "norm b Failed");

    //norm_b is calculated once
    T h_norm_b = norm_b[0];

    if (h_norm_b <= std::numeric_limits<T>::min())
    {
        solverControl->nIters = 0;
        solverControl->absoluteTolerance = 0.0;
        solverControl->relativeTolerance = 0.0;
        //we can either fill the x with zeros or cpy b to x;
        x = b;
        return clsparseSuccess;
    }



    //n == number of rows;
    const auto N = pA->n;

    vector y   (control, N, 0, CL_MEM_READ_WRITE, false);
    vector p   (control, N, 0, CL_MEM_READ_WRITE, false);
    vector r   (control, N, 0, CL_MEM_READ_WRITE, false);
    vector r_star (control, N, 0, CL_MEM_READ_WRITE, false);
    vector s   (control, N, 0, CL_MEM_READ_WRITE, false);
    vector Mp  (control, N, 0, CL_MEM_READ_WRITE, false);
    vector AMp (control, N, 0, CL_MEM_READ_WRITE, false);
    vector Ms  (control, N, 0, CL_MEM_READ_WRITE, false);
    vector AMs (control, N, 0, CL_MEM_READ_WRITE, false);

    scalar one(control,  1, CL_MEM_READ_ONLY, true);
    scalar zero(control, 0, CL_MEM_READ_ONLY, true);

    // y = A * x
    status = csrmv<T>(one, pA, x, zero, y, control);
    OPENCL_V_THROW(status, "csrmv Failed");

    // r = b - y
    status = r.sub(b, y, control);
    OPENCL_V_THROW(status, "b - y Failed");

    scalar norm_r (control, 0, CL_MEM_WRITE_ONLY, false);
    status = Norm1<T>(norm_r, r, control);
    OPENCL_V_THROW(status, "norm r Failed");

    scalar residuum (control, 0, CL_MEM_WRITE_ONLY, false);
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

    // p = r
    p = r; //deep copy;

    //Choose an arbitrary vector r̂0 such that (r̂0, r0) ≠ 0, e.g., r̂0 = r0
    r_star = r; //deep copy

    // holder for <r_star, r>
    scalar r_star_r_old (control, 0, CL_MEM_WRITE_ONLY, false);
    // holder for <r_star, r_{i+1}>
    scalar r_star_r_new (control, 0, CL_MEM_WRITE_ONLY, false);

    status = dot<T>(r_star_r_old, r_star, r, control);
    OPENCL_V_THROW(status, "<r_star, r> Failed");

    int iteration = 0;
    bool converged = false;

    scalar alpha (control, 0, CL_MEM_READ_WRITE, false);
    scalar beta  (control, 0, CL_MEM_READ_WRITE, false);
    scalar omega (control, 0, CL_MEM_READ_WRITE, false);

    // holder for <r_star, AMp>
    scalar r_star_AMp(control, 0, CL_MEM_WRITE_ONLY, false);

    // hoder for <A*M*s, s>
    scalar AMsS(control, 0, CL_MEM_WRITE_ONLY, false);
    // holder for <AMs, AMs>
    scalar AMsAMs(control, 0, CL_MEM_WRITE_ONLY, false);

    // holder for norm_s;
    scalar norm_s(control, 0, CL_MEM_WRITE_ONLY, false);

    while (!converged)
    {

        //Mp = M*p //apply preconditioner
        M(p, Mp, control);

        //AMp = A*Mp
        status = csrmv<T>(one, pA, Mp, zero, AMp, control);
        OPENCL_V_THROW(status, "csrmv A*Mp Failed");

        //<r_star, A*M*p>
        status = dot<T>(r_star_AMp, r_star, AMp, control);
        OPENCL_V_THROW(status, "<r_star, A*M*p> Failed");

        status = alpha.div(r_star_r_old, r_star_AMp, control);
        OPENCL_V_THROW(status, "alpha.div Failed");

        //s_j = r - alpha*Mp
        status = axpby<T, EW_MINUS>(s, one, r, alpha, AMp, control);
        OPENCL_V_THROW(status, "s_j = r - alpha * Mp Failed");

        status = Norm1<T>(norm_s, s, control);
        OPENCL_V_THROW(status, "norm s Failed");

        residuum.div(norm_s, norm_b, control);
        if (solverControl->finished(residuum[0]))
        {
//            iteration++;
            solverControl->nIters = iteration;
            //x = x + alpha * M*p_j;
            status = axpby<T>(x, one, x, alpha, Mp, control);
            OPENCL_V_THROW(status, "x = x + alpha * M*p Failed");
            break;
        }

        //Ms = M*s
        M(s, Ms, control);

        //AMs = A*Ms
        status = csrmv<T>(one, pA, Ms, zero, AMs, control);
        OPENCL_V_THROW(status, "csrmv AMs = A*Ms Failed");

        status = dot<T>(AMsS, AMs, s, control);
        OPENCL_V_THROW(status, "<AMs, s> Failed");

        status = dot<T> (AMsAMs, AMs, AMs, control);
        OPENCL_V_THROW(status, "<AMs, AMs> Failed");
        omega.div(AMsS, AMsAMs, control);

#ifndef NDEBUG
        if(omega[0] == 0)
            std::cout << "omega = 0" ;
#endif

        //x = x + alpha*Mp + omega*Ms;
        status = axpy<T>(x, alpha, Mp, x, control);
        OPENCL_V_THROW(status, "x = x + alpha * Mp Failed");

        status = axpy<T>(x, omega, Ms, x, control);
        OPENCL_V_THROW(status, "x = x + omega * Ms Failed");

        // r = s - omega * A*M*s
        status = axpy<T, EW_MINUS>(r, omega, AMs, s, control);
        OPENCL_V_THROW(status, "r = s - omega * AMs Failed");

        status = Norm1<T>(norm_r, r, control);
        OPENCL_V_THROW(status, "norm r Failed");

        status = residuum.div(norm_r, norm_b, control);
        OPENCL_V_THROW(status, "residuum = norm_r / norm_b Failed");

        if (solverControl->finished(residuum[0]))
        {
//            iteration++;
            solverControl->nIters = iteration;
            break;
        }

        //beta = <r_star, r+1> / <r_star, r> * (alpha/omega)
        status = dot<T>(r_star_r_new, r_star, r, control);
        OPENCL_V_THROW(status, "<r_star, r+1> Failed");

        //TODO:: is it the best order?
        status = beta.div(r_star_r_new, r_star_r_old, control);
        OPENCL_V_THROW(status, "<r_star, r+1> Failed");
        status = beta.mul(alpha, control);
        OPENCL_V_THROW(status, " b *= alpha Failed");
        status = beta.div(omega, control);
        OPENCL_V_THROW(status, "b/=omega Failed");

        r_star_r_old = r_star_r_new;

        //p = r + beta* (p - omega A*M*p);
        status = axpy<T>(p, beta, p, r, control); // p = beta * p + r;
        OPENCL_V_THROW(status, "p = beta*p + r Failed");
        status = beta.mul(omega, control);  // (beta*omega)
        OPENCL_V_THROW(status, "beta*=omega Failed");
        status = axpy<T,EW_MINUS>(p, beta, AMp, p, control);  // p = p - beta*omega*AMp;
        OPENCL_V_THROW(status, "p = p - beta*omega*AMp Failed");

        iteration++;
        solverControl->nIters = iteration;

        solverControl->print();
    }

}

#endif //SOLVER_BICGSTAB_HPP_
