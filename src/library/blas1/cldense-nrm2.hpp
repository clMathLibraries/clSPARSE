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
#ifndef _CLSPARSE_NRM_2_HPP_
#define _CLSPARSE_NRM_2_HPP_

#include "reduce.hpp"

template<typename T>
clsparseStatus
Norm2(clsparseScalarPrivate* pS,
      const cldenseVectorPrivate* pX,
      const clsparseControl control)
{
    return reduce<T, RO_SQR, RO_SQRT>(pS, pX, control);
}

/*
 * clsparse::array
 */
template<typename T>
clsparseStatus
Norm2(clsparse::vector<T>& pS,
      const clsparse::vector<T>& pX,
      const clsparseControl control)
{
    return reduce<T, RO_SQR, RO_SQRT>(pS, pX, control);
}


#endif //_CLSPARSE_NRM_2_HPP_
