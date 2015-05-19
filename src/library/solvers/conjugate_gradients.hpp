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



template<typename T>
clsparseStatus
cg(clsparseVector *x,
   const clsparseCsrMatrix* A,
   const clsparseVector *b,
   clSParseSolverControl solverControl,
   clsparseControl control)
{
    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    if (solverControl == nullptr)
    {
        return clsparseInvalidSolverControlObject;
    }

    clsparseVectorPrivate* pX = static_cast<clsparseVectorPrivate*> ( x );
    const clsparseCsrMatrixPrivate* pA = static_cast<const clsparseCsrMatrixPrivate*> ( A );
    const clsparseVectorPrivate* pB = static_cast<const clsparseVectorPrivate*> ( b );


    assert (pA->n == pB->n);
    assert (pA->m == pX->n);
    if ( (pA->n != pB->n) || (pA->m != pX->n) )
    {
        return clsparseInvalidSystemSize;
    }

    std::shared_ptr<PreconditionerHandler<T>> preconditioner;


    if (solverControl->preconditioner == DIAGONAL)
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new DiagonalHandler<T>());
        // call constructor of preconditioner class
        preconditioner->notify(pA, control);
    }
    else
    {
        preconditioner = std::shared_ptr<PreconditionerHandler<T>>(new VoidHandler<T>());
        preconditioner->notify(pA, control);
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

    {
        clMemRAII<T> m_norm_b(control->queue(), norm_b.value);
        T* f_norm_b = m_norm_b.clMapMem(CL_TRUE, CL_MAP_READ, 0, 1);

        std::cout << "norm_b " << *f_norm_b << std::endl;

        if (*f_norm_b == 0) //special case b is zero so solution is x = 0
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





    return clsparseSuccess;

}

#endif //_CLSPARSE_SOLVER_CG_HPP_
