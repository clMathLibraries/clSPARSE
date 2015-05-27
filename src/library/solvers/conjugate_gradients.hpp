#pragma #once
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
#define CLSP_ERRCHK(ans) { clsparseCheck((ans), __FILE__, __LINE__); }

inline void clsparseCheck(cl_int code, const char *file, int line, bool abort=false)
{
   if (code != CL_SUCCESS)
   {
      std::cerr << "clsparse assert: " << code << " " << file << " " << line << std::endl;
      if (abort) exit(code);
   }
}


// To keep the code more clear at the moment I've provided those functions for
// copying functions for vector and scalars
template <typename T>
inline cl_int
clsparseCpyVectorBuffers(const clsparseVectorPrivate* src,
                         clsparseVectorPrivate* dst,
                         clsparseControl control)
{
    cl_int status;
#if (BUILD_CLVERSION < 200)
            status = clEnqueueCopyBuffer(control->queue(), src->values, dst->values,
                            src->offset(), dst->offset(),
                            src->n * sizeof(T),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( )
                            );
#else
            status clEnqueueSVMMemcpy(control->queue(), CL_TRUE,
                           dst->values, src->values, src->n * sizeof(T),
                           control->event_wait_list.size(),
                           &(control->event_wait_list.front())(),
                           &control->event( ));
#endif
    return status;
}

template <typename T>
inline cl_int
clsparseCpyScalarBuffers(const clsparseScalarPrivate* src,
                         clsparseScalarPrivate* dst,
                         clsparseControl control)
{
    cl_int status;
#if (BUILD_CLVERSION < 200)
            status = clEnqueueCopyBuffer(control->queue(), src->value, dst->value,
                            src->offset(), dst->offset(),
                            sizeof(T),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( )
                            );
#else
            status clEnqueueSVMMemcpy(control->queue(), CL_TRUE,
                           dst->value, src->value, sizeof(T),
                           control->event_wait_list.size(),
                           &(control->event_wait_list.front())(),
                           &control->event( ));
#endif
    return status;
}

template <typename T>
inline cl_int
clsparse_alloc_init_vector(clsparseVectorPrivate& vec, T value, clsparseControl control, bool fill = true)
{
    cl_int status;
    cl_context ctx = control->getContext()();

    vec.values = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(T) * vec.n, NULL, &status);
    CLSP_ERRCHK(status);

    if (fill)
    {
        status = clEnqueueFillBuffer(control->queue(), vec.values, &value, sizeof(T), 0,
                            sizeof(T) * vec.n, 0, NULL, NULL);
        CLSP_ERRCHK(status);
    }

    return status;
}

template <typename T>
inline cl_int
clsparse_alloc_init_scalar(clsparseScalarPrivate& scalar, T value, clsparseControl control, bool fill = true)
{
    cl_int status;
    cl_context ctx = control->getContext()();

    scalar.value = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(T), NULL, &status);
    CLSP_ERRCHK(status);

    if (fill)
    {
        status = clEnqueueFillBuffer(control->queue(), scalar.value, &value, sizeof(T), 0,
                            sizeof(T), 0, NULL, NULL);
        CLSP_ERRCHK(status);
    }
    return status;
}

template <typename T>
inline cl_int
display_vector(const clsparseVectorPrivate& v, clsparseControl control, std::string info = "")
{
    {
        size_t size = v.n;
        clMemRAII<T> r_temp(control->queue(), v.values);
        T* f_temp = r_temp.clMapMem(CL_TRUE, CL_MAP_READ, 0, size);

        std::cout << info << std::endl;
        for (int i = 0; i < 5; i++)
        {
            std::cout << f_temp[i] << std::endl;
        }
        std::cout << "..." << std::endl;
        for (int i = size - 5; i < size; i++)
        {
            std::cout << f_temp[i] << std::endl;
        }
    }

}

template <typename T>
inline cl_int
display_scalar(const clsparseScalarPrivate& s, clsparseControl control, std::string info = "")
{
    {
        size_t size = 1;
        clMemRAII<T> r_temp(control->queue(), s.value);
        T* f_temp = r_temp.clMapMem(CL_TRUE, CL_MAP_READ, 0, size);

        std::cout << info << " = " << *f_temp << std::endl;
    }
}

template <typename T>
inline cl_int
display_matrix(const clsparseCsrMatrix& m, clsparseControl control)
{
    clMemRAII<T> r_v(control->queue(), m.values);
    T* f_v = r_v.clMapMem(CL_TRUE, CL_MAP_READ, 0, m.nnz);

    for (int i = 0; i < m.nnz; i++)
    {
        std::cout << f_v[i] << std::endl;
    }
}





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
    cl_int status;

    T scalarOne = 1;
    T scalarZero = 0;

    cl_context ctx = control->getContext()();

//    clsparse::array<T> x(control, 10, 1.2);
//    std::cout << x[0] << std::endl;
//    x.fill(control, 1.4);
//    std::cout << x[0] << std::endl;
//    x[9] = 1.555;
//    std::cout << x[9] << std::endl;

//    clsparse::array<T> x2(control, 10, 0.1);
//    x2 = x.copy(control);
//    std::cout << "X2 content: " << std::endl;
//    for(int i = 0; i < x2.size(); i++)
//    {
//        std::cout << x2[i] << std::endl;
//    }

    clsparseScalarPrivate norm_b;
    clsparseInitScalar(&norm_b);
    status = clsparse_alloc_init_scalar(norm_b, scalarZero, control, false);
    CLSP_ERRCHK(status);


    //norm of rhs of equation
    status = Norm1<T>(&norm_b, pB, control);
    CLSP_ERRCHK(status);

    //norm_b is calculated once
    T h_norm_b = 0;

    //we do not have explicit unmap function defined so I'm doing this in that way
    {
        clMemRAII<T> m_norm_b(control->queue(), norm_b.value);
        T* f_norm_b = m_norm_b.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);
        h_norm_b = *f_norm_b;

#ifndef NDEBUG
        std::cout << "norm_b " << h_norm_b << std::endl;
#endif

        if (h_norm_b == 0) //special case b is zero so solution is x = 0
        {
            solverControl->nIters = 0;
            solverControl->absoluteTolerance = 0.0;
            solverControl->relativeTolerance = 0.0;

            //we can either fill the x with zeros or cpy b to x;
            status = clsparseCpyVectorBuffers<T>(pB, pX, control);
            CLSP_ERRCHK(status);
            return clsparseSuccess;
        }
    }

    //continuing "normal" execution of cg algorithm
    const auto N = pA->n;

    //helper containers, all need to be zeroed
    clsparseVectorPrivate y;
    clsparseInitVector(&y);
    y.n = N;
    status = clsparse_alloc_init_vector(y, scalarZero, control);
    CLSP_ERRCHK(status);


    clsparseVectorPrivate z;
    clsparseInitVector(&z);
    z.n = N;
    status = clsparse_alloc_init_vector(z, scalarZero, control);
    CLSP_ERRCHK(status);


    clsparseVectorPrivate r;
    clsparseInitVector(&r);
    r.n = N;
    status = clsparse_alloc_init_vector(r, scalarZero, control, false);
    CLSP_ERRCHK(status);


    clsparseVectorPrivate p;
    clsparseInitVector(&p);
    p.n = N;
    status = clsparse_alloc_init_vector(p, scalarZero, control, false);
    CLSP_ERRCHK(status);

    //TODO: Change sAlpha to one, bEta to 0;?
    clsparseScalarPrivate sAlpha;
    clsparseInitScalar(&sAlpha);
    status = clsparse_alloc_init_scalar(sAlpha, scalarOne, control);
    CLSP_ERRCHK(status);


    clsparseScalarPrivate sBeta;
    clsparseInitScalar(&sBeta);
    status = clsparse_alloc_init_scalar(sBeta, scalarZero, control);
    CLSP_ERRCHK(status);

    // y = A*x
    status = csrmv<T>(&sAlpha, pA, pX, &sBeta, &y, control);
    CLSP_ERRCHK(status);

    //r = b - y
    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);
    CLSP_ERRCHK(status);

    clsparseScalarPrivate norm_r;
    clsparseInitScalar(&norm_r);
    clsparse_alloc_init_scalar(norm_r, scalarZero, control, false);

    //calculate norm of r
    status = Norm1<T>(&norm_r, &r, control);
    CLSP_ERRCHK(status);

    T residuum = 0;
    {

        clMemRAII<T> m_norm_r(control->queue(), norm_r.value);
        T* f_norm_r = m_norm_r.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

        residuum = *f_norm_r / h_norm_b;
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
    M(&r, &z, control);

    //copy inital z to p
    status = clsparseCpyVectorBuffers<T>(&z, &p, control);
    CLSP_ERRCHK(status);

    //rz = <r, z>, here actually should be conjugate(r)) but we do not support complex type.
    clsparseScalarPrivate rz;
    clsparseInitScalar(&rz);
    status = clsparse_alloc_init_scalar(rz, scalarZero, control, false);
    CLSP_ERRCHK(status);

    status = dot<T>(&rz, &r, &z, control);
    CLSP_ERRCHK(status);

    int iteration = 0;

    bool converged = false;

    clsparseScalarPrivate alpha;
    clsparseInitScalar(&alpha);
    clMemRAII<T> m_alpha(control->queue(), &alpha.value, 1);

    clsparseScalarPrivate beta;
    clsparseInitScalar(&beta);
    clMemRAII<T> m_beta(control->queue(), &beta.value, 1);

    //yp buffer for inner product of y and p vectors;
    clsparseScalarPrivate yp;
    clsparseInitScalar(&yp);
    clMemRAII<T> m_yp(control->queue(), &yp.value, 1);

    clsparseScalarPrivate rz_old;
    clsparseInitScalar(&rz_old);
    clMemRAII<T> m_rz_old(control->queue(), &rz_old.value, 1);

    while(!converged)
    {
        solverControl->nIters = iteration;

        //y = A*p
        status = csrmv<T>(&sAlpha, pA, &p, &sBeta, &y, control);
        CLSP_ERRCHK(status);


        status = dot<T>(&yp, &y, &p, control);
        CLSP_ERRCHK(status);

        // alpha = <r,z> / <y,p>
        {
            clMemRAII<T> r_alpha(control->queue(), alpha.value);
            T* f_alpha = r_alpha.clMapMem(CL_TRUE, CL_MAP_WRITE, 0, 1);

            clMemRAII<T> r_rz(control->queue(), rz.value);
            T* f_rz = r_rz.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

            clMemRAII<T> r_yp(control->queue(), yp.value);
            T* f_yp = r_yp.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

            *f_alpha = *f_rz / *f_yp;
#ifndef NDEBUG
            std::cout << "alpha = " << *f_alpha << std::endl;
#endif
        }

        //x = x + alpha*p
        status = axpy<T>(pX->n, pX, &alpha, &p, control);
        CLSP_ERRCHK(status);

        //r = r - alpha * y;
        status = axpy<T, EW_MINUS>(r.n, &r, &alpha, &y, control);
        CLSP_ERRCHK(status);

        //apply preconditioner z = M*r
        M(&r, &z, control);

        //store old value of rz
        status = clsparseCpyScalarBuffers<T>(&rz, &rz_old, control);
        CLSP_ERRCHK(status);

        //rz = <r,z>
        status = dot<T>(&rz, &r, &z, control);
        CLSP_ERRCHK(status);

        // beta = <r^(i), r^(i)>/<r^(i-1),r^(i-1)> // i: iteration index;
        // beta is ratio of dot product in current iteration compared
        // to previous.
        {
            clMemRAII<T> r_beta(control->queue(), beta.value);
            T* f_beta = r_beta.clMapMem(CL_TRUE, CL_MAP_WRITE, 0, 1);

            clMemRAII<T> r_rz(control->queue(), rz.value);
            T* f_rz = r_rz.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);


            clMemRAII<T> r_rz_old(control->queue(), rz_old.value);
            T* f_rz_old = r_rz_old.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

            *f_beta = *f_rz / *f_rz_old;
#ifndef NDEBUG
            std::cout << "beta = " << *f_beta << std::endl;
#endif
        }

        //p = z + beta*p;
        //TODO: change name sAlpha to "one"
        status = axpby<T>(p.n, &p, &sAlpha, &z, &beta, control );
        CLSP_ERRCHK(status);

        //calculate norm of r
        status = Norm1<T>(&norm_r, &r, control);
        CLSP_ERRCHK(status);

        {

            clMemRAII<T> m_norm_r(control->queue(), norm_r.value);
            T* f_norm_r = m_norm_r.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

            residuum = *f_norm_r / h_norm_b;

        }

        iteration++;
        converged = solverControl->finished(residuum);

        solverControl->print();

    }

    clReleaseMemObject(norm_b.value);
    clReleaseMemObject(y.values);
    clReleaseMemObject(z.values);
    clReleaseMemObject(r.values);
    clReleaseMemObject(p.values);

    clReleaseMemObject(sAlpha.value);
    clReleaseMemObject(sBeta.value);

    clReleaseMemObject(norm_r.value);
    clReleaseMemObject(rz.value);

    return clsparseSuccess;
}

#endif //_CLSPARSE_SOLVER_CG_HPP_
