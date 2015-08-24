#include "reduce-by-key.hpp"


clsparseStatus
clsparseSreduceByKey(cldenseVector* keys_out,
                     cldenseVector* vals_out,
                     const cldenseVector* keys_in,
                     const cldenseVector* vals_in,
                     const clsparseControl control)
{
    clsparse::vector<int> KeysIn (control, keys_in->values, keys_in->num_values);
    clsparse::vector<int> KeysOut (control, keys_out->values, keys_out->num_values);

    clsparse::vector<cl_float> ValsIn(control, vals_in->values, vals_in->num_values);
    clsparse::vector<cl_float> ValsOut(control, vals_out->values, vals_out->num_values);


    return internal::reduce_by_key(KeysOut, ValsOut, KeysIn, ValsIn, control);

}

clsparseStatus
clsparseDreduceByKey(cldenseVector* keys_out,
                     cldenseVector* vals_out,
                     const cldenseVector* keys_in,
                     const cldenseVector* vals_in,
                     const clsparseControl control)
{
    clsparse::vector<int> KeysIn (control, keys_in->values, keys_in->num_values);
    clsparse::vector<int> KeysOut (control, keys_out->values, keys_out->num_values);

    clsparse::vector<cl_double> ValsIn(control, vals_in->values, vals_in->num_values);
    clsparse::vector<cl_double> ValsOut(control, vals_out->values, vals_out->num_values);


    return internal::reduce_by_key(KeysOut, ValsOut, KeysIn, ValsIn, control);

}
