#pragma once
#ifndef _CLSPARSE_PRECONDITIONER_HPP_
#define _CLSPARSE_PRECONDITIONER_HPP_

#include "include/clSPARSE-private.hpp"
//enum PRECONDITIONER
//{
//    VOID = 0,
//    DIAGONAL
//};

template<typename T>
class PreconditionerHandler
{
public:
    virtual void operator()(const clsparseVectorPrivate* x,
                       clsparseVectorPrivate* y,
                       clsparseControl control) = 0;

    virtual void notify(const clsparseCsrMatrixPrivate* pA,
                        clsparseControl control) = 0;
};

#endif //_CLSPARSE_PRECONDITIONER_HPP_
