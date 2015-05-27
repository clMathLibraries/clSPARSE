#pragma once
#ifndef _CLSPARSE_NRM_1_HPP_
#define _CLSPARSE_NRM_1_HPP_

#include "internal/data_types/clarray.hpp"
#include "reduce.hpp"


template<typename T>
clsparseStatus
Norm1(clsparseScalarPrivate* pS,
      const clsparseVectorPrivate* pX,
      const clsparseControl control)
{
    return reduce<T, RO_FABS>(pS, pX, control);
}

#endif //_CLSPARSE_NRM_1_HPP_
