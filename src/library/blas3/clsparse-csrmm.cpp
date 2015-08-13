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

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-internal.hpp"
#include "internal/clsparse-control.hpp"
#include "clsparse-csrmm.hpp"

clsparseStatus
clsparseScsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* sparseCsrA,
const cldenseMatrix* denseB,
const clsparseScalar* beta,
cldenseMatrix* denseC,
const clsparseControl control )
{
    if( !clsparseInitialized )
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if( control == nullptr )
    {
        return clsparseInvalidControlObject;
    }

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pSparseCsrA = static_cast<const clsparseCsrMatrixPrivate*>( sparseCsrA );
    const cldenseMatrixPrivate* pDenseB = static_cast<const cldenseMatrixPrivate*>( denseB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pDenseC = static_cast<cldenseMatrixPrivate*>( denseC );

    return csrmm< cl_float >( *pAlpha, *pSparseCsrA, *pDenseB, *pBeta, *pDenseC, control );
}

clsparseStatus
clsparseDcsrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* sparseCsrA,
const cldenseMatrix* denseB,
const clsparseScalar* beta,
cldenseMatrix* denseC,
const clsparseControl control )
{
    if( !clsparseInitialized )
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if( control == nullptr )
    {
        return clsparseInvalidControlObject;
    }

    const clsparseScalarPrivate* pAlpha = static_cast<const clsparseScalarPrivate*>( alpha );
    const clsparseCsrMatrixPrivate* pSparseCsrA = static_cast<const clsparseCsrMatrixPrivate*>( sparseCsrA );
    const cldenseMatrixPrivate* pDenseB = static_cast<const cldenseMatrixPrivate*>( denseB );
    const clsparseScalarPrivate* pBeta = static_cast<const clsparseScalarPrivate*>( beta );
    cldenseMatrixPrivate* pDenseC = static_cast<cldenseMatrixPrivate*>( denseC );

    return csrmm< cl_double >( *pAlpha, *pSparseCsrA, *pDenseB, *pBeta, *pDenseC, control );

}
