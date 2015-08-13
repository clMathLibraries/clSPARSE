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
#ifndef _CLSPARSE_COMMONS_HPP_
#define _CLSPARSE_COMMONS_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/data-types/clarray-base.hpp"
template <typename T>
inline void init_scalar(clsparseScalarPrivate* scalar, T value,
                        const clsparseControl control)
{
    clMemRAII<T> rScalar (control->queue(), scalar->value);

    T* fR = rScalar.clMapMem( CL_TRUE, CL_MAP_WRITE, scalar->offset(), 1);

    *fR  = value;
}

template <typename T>
inline void init_scalar(clsparse::array_base<T>& scalar, T value,
                        const clsparseControl control)
{
    clsparse::internal::fill(scalar, value, control->queue);
}


//    vector.values = ::clSVMAlloc(control->getContext()(), CL_MEM_READ_WRITE,
//                                 size * sizeof(T), 0);
//    ::clSVMFree(control->getContext()(), vector.values)

#endif //_CLSPARSE_COMMONS_HPP_
