#pragma once
#ifndef _CLSPARSE_COMMONS_HPP_
#define _CLSPARSE_COMMONS_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/data_types/clarray_base.hpp"
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
