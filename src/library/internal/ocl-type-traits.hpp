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

#ifndef OCL_TYPE_TRAITS_HPP_
#define OCL_TYPE_TRAITS_HPP_

#ifdef __ALTIVEC__
#include <altivec.h>
#undef bool
#undef vector
#endif

#include <type_traits>

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DECLARE_TYPE(TYPE) template<> struct OclTypeTraits<TYPE> \
{ static const char* type;};

template<typename T>
struct OclTypeTraits
{
};

DECLARE_TYPE( cl_char )
DECLARE_TYPE( cl_uchar )
DECLARE_TYPE( cl_short )
DECLARE_TYPE( cl_ushort )
DECLARE_TYPE( cl_int )
DECLARE_TYPE( cl_uint )
DECLARE_TYPE( cl_long )
DECLARE_TYPE( cl_ulong )
DECLARE_TYPE( cl_float )
DECLARE_TYPE( cl_double )


//cl_mem is pointer to non fundamental type _cl_mem
//is_clmem returns true for T = cl_mem
template <typename T>
struct is_pointer_fundamental
{
    static bool const value =
        (std::is_pointer<T>::value &&
        std::is_fundamental<typename std::remove_pointer<T>::type>::value);
};


#endif // OCL_TYPE_TRAITS_HPP_
