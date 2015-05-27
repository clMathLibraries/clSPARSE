#pragma once
#ifndef _CLSPARSE_PREC_DIAGONAL_HPP_
#define _CLSPARSE_PREC_DIAGONAL_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise_transform.hpp"
#include "preconditioner_utils.hpp"
#include <memory>
#include <algorithm>
/* The simplest preconditioner consists of just the
   inverse values of the diagonal of the matrix:

   The Jacobi preconditioner is one of the simplest forms of preconditioning,
   in which the preconditioner is chosen to be the diagonal of the matrix

                        P = \mathrm{diag}(A).

    Assuming A_{ii} \neq 0, \forall i ,

    we get P^{-1}_{ij} = \frac{\delta_{ij}}{A_{ij}}.

    It is efficient for diagonally dominant matrices A.
*/

template<typename T>
class DiagonalPreconditioner
{
public:
    DiagonalPreconditioner(const clsparseCsrMatrixPrivate* A,
                           clsparseControl control)
    {
        //allocate proper size assuming rectangular size of A;
        cl_uint size = min(A->m, A->n);

        clsparseInitVector(&invDiag_A);
        invDiag_A.n = size;

        cl_int status;

        clMemRAII<T> r_invDiag_A (control->queue(),
                                  &invDiag_A.values, invDiag_A.n,
                                                CL_MEM_READ_WRITE);

        // I dont want to clMemRAII release this object in constructor
        // but the problem will be in openCL 20
        // write operator= for clMemRAII then extend this class with field
        // of this type which will manage the invDiag_A.
#if (BUILD_CLVERSION < 200)
        status = ::clRetainMemObject(invDiag_A.values);
#endif

        if( status != CL_SUCCESS )
        {
            std::cout << "Problem with creating invDiag buffer"
                      << " (" << status << ")" << std::endl;
        }
        // extract inverse diagonal from matrix A and store it in invDiag_A
        // easy to check with poisson matrix;
        status = extract_diagonal<T, true>(&invDiag_A, A, control);

        if( status != CL_SUCCESS )
        {
            std::cout << "Invalid extract_diagonal kernel execution " << std::endl;
        }
        //Print the values from invDiag_A
//        else
//        {
//            //clMemRAII<T> rData (control->queue(), invDiag_A.values);
//            T* data = r_invDiag_A.clMapMem(CL_TRUE, CL_MAP_READ, invDiag_A.offset(), invDiag_A.n);

//            for (int i = 0; i < invDiag_A.n; i++)
//            {
//                std::cout << "i = " << i << " " << data[i] << std::endl;
//            }
//            std::cout << std::endl;
//        }

    }

    // apply preconditioner
    void operator ()(const clsparseVectorPrivate* x,
                     clsparseVectorPrivate* y,
                     clsparseControl control)
    {
        //element wise multiply y = x*invDiag_A;
        clsparseStatus status =
                elementwise_transform<T, EW_MULTIPLY>(y, x, &invDiag_A, control);
    }

    ~DiagonalPreconditioner()
    {
        ::clReleaseMemObject(invDiag_A.values);
        // return to init state;
        clsparseInitVector(&invDiag_A);
    }

private:
    //inverse diagonal values of matrix A;
    clsparseVectorPrivate invDiag_A;
};


template<typename T>
class DiagonalHandler : public PreconditionerHandler<T>
{
public:

    using Diag = DiagonalPreconditioner<T>;

    DiagonalHandler()
    {
    }

    void operator()(const clsparseVectorPrivate* x,
                    clsparseVectorPrivate* y,
                    clsparseControl control)
    {
        (*diagonal)(x, y, control);
    }

    void notify(const clsparseCsrMatrixPrivate* pA, clsparseControl control)
    {
        diagonal = std::make_shared<Diag>(pA, control);
    }

private:
    std::shared_ptr<Diag> diagonal;
};

#endif //_CLSPARSE_PREC_DIAGONAL_HPP_
