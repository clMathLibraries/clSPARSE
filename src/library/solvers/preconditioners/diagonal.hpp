#pragma once
#ifndef _CLSPARSE_PREC_DIAGONAL_HPP_
#define _CLSPARSE_PREC_DIAGONAL_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "preconditioner.hpp"

#include "blas1/elementwise_transform.hpp"
#include "preconditioner_utils.hpp"
#include <memory>

template<typename T>
class Diagonal
{
public:
    Diagonal(const clsparseCsrMatrixPrivate* A, clsparseControl control)
    {
        //allocate proper size assuming rectangular size of A;
        cl_uint size = std::min(A->m, A->n);

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
        ::clRetainMemObject(invDiag_A.values);
#endif

        if( status != CL_SUCCESS )
        {
            std::cout << "Problem with creating invDiag buffer" << std::endl;
        }
        status = extract_diagonal<T, false>(&invDiag_A, A, control);

        if( status != CL_SUCCESS )
        {
            std::cout << "Invalid extract_diagonal kernel execution " << std::endl;
        }
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

    void operator ()(const clsparseVectorPrivate* x,
                     clsparseVectorPrivate* y,
                     clsparseControl control)
    {
        //element wise multiply x*invDiag_A ---> y;
        clsparseStatus status =
                elementwise_transform<T, MULTIPLY>(y, x, &invDiag_A, control);
    }

    ~Diagonal()
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
class PrecondDiagonalHandler : public PreconditionerHandler<T>
{
public:

    using Diag = Diagonal<T>;

    PrecondDiagonalHandler()
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
