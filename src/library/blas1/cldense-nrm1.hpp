/* ************************************************************************
 * Copyright 2015 Vratis, Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ************************************************************************ */

#pragma once
#ifndef _CLSPARSE_NRM_1_HPP_
#define _CLSPARSE_NRM_1_HPP_

#include "internal/data-types/clvector.hpp"
#include "reduce.hpp"


template<typename T>
clsparseStatus
Norm1(clsparseScalarPrivate* pS,
      const cldenseVectorPrivate* pX,
      const clsparseControl control)
{
    return reduce<T, RO_FABS>(pS, pX, control);
}

/*
 * clsparse::array
 */
template<typename T>
clsparseStatus
Norm1(clsparse::array_base<T>& pS,
      const clsparse::array_base<T>& pX,
      const clsparseControl control)
{
    return reduce<T, RO_FABS>(pS, pX, control);
}

#endif //_CLSPARSE_NRM_1_HPP_
