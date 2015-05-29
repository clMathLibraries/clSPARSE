#pragma once
#ifndef _CLSPARSE_CSRMV_HPP_
#define _CLSPARSE_CSRMV_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse_control.hpp"
#include "spmv/csrmv_adaptive/csrmv_adaptive.hpp"
#include "spmv/csrmv_vector/csrmv_vector.hpp"


template <typename T>
clsparseStatus
csrmm( const clsparseScalar* alpha,
const clsparseCsrMatrix* matA,
const cldenseMatrix* matB,
const clsparseScalar* beta,
cldenseMatrix* matC,
const clsparseControl control )
{

}

#endif
