/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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
/*! \file
 *  \brief cufunc_sparse-xx.h defines generic type for 32/64 bit indices
 */

#pragma once
#ifndef _CU_SPARSE_xx_H_
#define _CU_SPARSE_xx_H_

#if( CLSPARSE_INDEX_SIZEOF == 8 )
#error Wait till clSPARSE implements 64-bit indices
   typedef unsigned long long clsparseIdx_t;
#else
   typedef int clsparseIdx_t;
#endif

#if( CLSPARSE_INDEX_SIZEOF == 8 )
#define SIZET "l"
#else
#define SIZET ""
#endif


#endif // ifndef  _CU_SPARSE_xx_H_
