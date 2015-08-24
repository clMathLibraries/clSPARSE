
#include "clSPARSE.h"
#include "internal/data-types/clvector.hpp"

#include "scan.hpp"

template<typename T>
clsparseStatus
clsparseScan(clsparse::vector<T>& Output,
             const clsparse::vector<T>& Input,
             int exclusive,
             const clsparseControl control)
{
    if (exclusive == 1)
        return exclusive_scan<EW_PLUS>(Output, Input, control);
    else if (exclusive == 0 )
        return inclusive_scan<EW_PLUS>(Output, Input, control);
    else
        return clsparseInvalidValue;


}

clsparseStatus
clsparseIscan(cldenseVector* r,
              const cldenseVector* x,
              int exclusive,
              const clsparseControl control)
{
    clsparse::vector<cl_int> Output(control, r->values, r->num_values);
    clsparse::vector<cl_int> Input(control, x->values, x->num_values);


    return clsparseScan(Output, Input, exclusive, control);

}

clsparseStatus
clsparseSscan(cldenseVector* r,
              const cldenseVector* x,
              int exclusive,
              const clsparseControl control)
{
    clsparse::vector<cl_float> Output(control, r->values, r->num_values);
    clsparse::vector<cl_float> Input(control, x->values, x->num_values);

    return clsparseScan(Output, Input, exclusive, control);

}

clsparseStatus
clsparseDscan(cldenseVector* r,
              const cldenseVector* x,
              int exclusive,
              const clsparseControl control)
{
    clsparse::vector<cl_double> Output(control, r->values, r->num_values);
    clsparse::vector<cl_double> Input(control, x->values, x->num_values);


    return clsparseScan(Output, Input, exclusive, control);

}

