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
#ifndef _CLSPARSE_CSRMV_HPP_
#define _CLSPARSE_CSRMV_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"
#include "blas2/csrmv-adaptive.hpp"
#include "blas2/csrmv-vector.hpp"
#include "internal/data-types/clvector.hpp"

template <typename T>
clsparseStatus
csrmv (const clsparseScalarPrivate *pAlpha,
       const clsparseCsrMatrixPrivate *pCsrMatx,
       const cldenseVectorPrivate *pX,
       const clsparseScalarPrivate *pBeta,
       cldenseVectorPrivate *pY,
       clsparseControl control)
{
    if( pCsrMatx->meta == nullptr )
    {
        return csrmv_vector<T>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
    }
    else
    {
        const matrix_meta* meta_ptr = static_cast< const matrix_meta* >( pCsrMatx->meta );
        if( meta_ptr->rowBlockSize == 0 )
        {
            // rowBlockSize variable is not zero but no pointer
            return clsparseStructInvalid;
        }

        // Use this for csrmv_general instead of adaptive.
        //return csrmv_vector<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );

        // Call adaptive CSR kernels
        return csrmv_adaptive<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );

    }
}

/*
 * clsparse::array
 */

template <typename T>
clsparseStatus
csrmv (const clsparse::array_base<T>& pAlpha,
       const clsparseCsrMatrixPrivate *pCsrMatx,
       const clsparse::array_base<T>& pX,
       const clsparse::array_base<T>& pBeta,
       clsparse::array_base<T>& pY,
       clsparseControl control)
{
    if( pCsrMatx->meta == nullptr )
    {
        return csrmv_vector<T>(pAlpha, pCsrMatx, pX, pBeta, pY, control);
    }
    else
    {
        const matrix_meta* meta_ptr = static_cast< const matrix_meta* >( pCsrMatx->meta );
        if( meta_ptr->rowBlockSize == 0 )
        {
            // rowBlockSize variable is not zero but no pointer
            return clsparseStructInvalid;
        }

        // Use this for csrmv_general instead of adaptive.
        //return csrmv_vector<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );

        // Call adaptive CSR kernels
        return csrmv_adaptive<T>( pAlpha, pCsrMatx, pX, pBeta, pY, control );
    }

}

#endif
