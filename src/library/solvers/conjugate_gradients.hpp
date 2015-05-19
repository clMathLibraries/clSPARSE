#pragma #once
#ifndef _CLSPARSE_SOLVER_CG_HPP_
#define _CLSPARSE_SOLVER_CG_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_internal.hpp"

#include "preconditioners/preconditioner.hpp"
#include "preconditioners/diagonal.hpp"
#include "preconditioners/void.hpp"

//matrix multiply
#include "spmv/csrmv_vector/csrmv_vector_impl.hpp"
//dense vector operations
#include "blas1/reduce.hpp"

#include "solver_control.hpp"



#define CLSP_ERRCHK(ans) { clsparseCheck((ans), __FILE__, __LINE__); }

inline void clsparseCheck(cl_int code, const char *file, int line, bool abort=false)
{
   if (code != CL_SUCCESS)
   {
      std::cerr << "clsparse assert: " << code << " " << file << " " << line << std::endl;
      if (abort) exit(code);
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

    //start of the solver algorithm
    clsparseScalarPrivate norm_b;
    clsparseInitScalar(&norm_b);

    //let the clMemRAII control the norm_b object
    //TODO:: Implement Allocator which will control this object!
    clMemRAII<T> r_norm_b(control->queue(), &norm_b.value, 1);


    /*TODO:: create an internal header file which will provide a nice named
             functions like norm instead of reduce<T, RO_FABS>. Currently
             this names are defined in cpp files so can't be extracted
    */

    //norm of rhs
    cl_int status = reduce<T, RO_FABS>(&norm_b, pB, control);
    CLSP_ERRCHK(status);

    //norm_b is calculated once
    T h_norm_b = 0;
    //we do not have explicit unmap function defined so I'm doing this in that way
    {
        clMemRAII<T> m_norm_b(control->queue(), norm_b.value);
        T* f_norm_b = m_norm_b.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);
        h_norm_b = *f_norm_b;

        std::cout << "norm_b " << h_norm_b << std::endl;

        if (h_norm_b == 0) //special case b is zero so solution is x = 0
        {
            solverControl->nIters = 0;
            solverControl->absoluteTolerance = 0.0;
            solverControl->relativeTolerance = 0.0;

            //we can either fill the x with zeros or cpy b to x;
#if (BUILD_CLVERSION < 200)
            status = clEnqueueCopyBuffer(control->queue(), pB->values, pX->values,
                            pB->offset(), pX->offset(),
                            pB->n * sizeof(T),
                            control->event_wait_list.size(),
                            &(control->event_wait_list.front())(),
                            &control->event( )
                            );
#else
            status clEnqueueSVMMemcpy(control->queue(), CL_TRUE,
                           pX->values, pB->values, pB->n * sizeof(T),
                           control->event_wait_list.size(),
                           &(control->event_wait_list.front())(),
                           &control->event( ));
#endif
            CLSP_ERRCHK(status);
            std::cout << "vec B = 0" << std::endl;
            return clsparseSuccess;
        }
    }


    //continuing "normal" execution of cg algorithm

    const auto N = pA->n;

    //helper containers, all need to be zeroed
    clsparseVectorPrivate y;
    clsparseInitVector(&y);
    y.n = N;
    //TODO: allocator instead clMemRAII here and for others
    clMemRAII<T> gy(control->queue(), &y.values, y.n);
    gy.clFillMem(0, 0, y.n);
    /* Test of clFillMem

//    {
//        T* hg = gy.clMapMem(CL_TRUE, CL_MAP_READ, 0, y.n);
//        for (int i = 0; i < y.n; i++)
//        {
//            std::cout << hg[i] << std::endl;
//        }
//    }
    */

    clsparseVectorPrivate z;
    clsparseInitVector(&z);
    z.n = N;
    clMemRAII<T> gz(control->queue(), &z.values, z.n);
    gz.clFillMem(0, 0, z.n);

    clsparseVectorPrivate r;
    clsparseInitVector(&r);
    r.n = N;
    clMemRAII<T> gr(control->queue(), &r.values, r.n);
    gr.clFillMem(0, 0, r.n);

    clsparseVectorPrivate p;
    clsparseInitVector(&p);
    p.n = N;
    clMemRAII<T> gp(control->queue(), &p.values, p.n);
    gp.clFillMem(0, 0, p.n);

    clsparseScalarPrivate alpha;
    clsparseInitScalar(&alpha);
    clMemRAII<T> ga(control->queue(), &alpha.value, 1);
    ga.clFillMem(1, 0, 1); //set alpha to 1

    clsparseScalarPrivate beta;
    clsparseInitScalar(&beta);
    clMemRAII<T> gb(control->queue(), &beta.value, 1);
    gb.clFillMem(0, 0, 1); //set beta to 0


    // y = A*x
    status = csrmv<T>(&alpha, pA, pX, &beta, &y, control);
    CLSP_ERRCHK(status);

    //r = b - y
    status = elementwise_transform<T, EW_MINUS>(&r, pB, &y, control);
    CLSP_ERRCHK(status);

    clsparseScalarPrivate norm_r;
    clsparseInitScalar(&norm_r);

    clMemRAII<T> r_norm_r(control->queue(), &norm_r.value, 1);

    //calculate norm of r
    status = reduce<T, RO_FABS>(&norm_r, &r, control);
    CLSP_ERRCHK(status);

    T residuum = 0;
    {

        clMemRAII<T> m_norm_r(control->queue(), norm_r.value);
        T* f_norm_r = m_norm_r.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

        residuum = *f_norm_r / h_norm_b;
        std::cout << "initial residuum = " << residuum << std::endl;
    }

    solverControl->initialResidual = residuum;
    if (solverControl->finished(residuum))
    {
        solverControl->nIters = 0;
        return clsparseSuccess;
    }

    //apply preconditioner
    M(&r, &z, control);




    return clsparseSuccess;

}

#endif //_CLSPARSE_SOLVER_CG_HPP_
