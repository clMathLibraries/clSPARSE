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

#include "ocl-type-traits.hpp"


#define DECLARE_TYPE_STR(TYPE, TYPE_STR) \
    const char* OclTypeTraits<TYPE>::type = TYPE_STR;

DECLARE_TYPE_STR( cl_char, "char" )
DECLARE_TYPE_STR( cl_uchar, "uchar" )
DECLARE_TYPE_STR( cl_short, "short" )
DECLARE_TYPE_STR( cl_ushort, "ushort" )
DECLARE_TYPE_STR( cl_int, "int" )
DECLARE_TYPE_STR( cl_uint, "uint" )
DECLARE_TYPE_STR( cl_long, "long" )
DECLARE_TYPE_STR( cl_ulong, "ulong" )
DECLARE_TYPE_STR( cl_float, "float" )
DECLARE_TYPE_STR( cl_double, "double" )
