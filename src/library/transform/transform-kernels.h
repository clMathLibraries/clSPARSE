CLSPARSE_EXPORT clsparseStatus
scan( int first,
      int last,
      cl_mem input_array,
      cl_mem output_result,
      int init_T,
      int exclusive,
      clsparseControl control
     );

CLSPARSE_EXPORT clsparseStatus
reduce_by_key(
      int keys_first,
      int keys_last,
      int values_first,
      cl_mem keys_input,
      cl_mem values_input,
      cl_mem keys_output,
      cl_mem values_output,
      int *count,
      clsparseControl control
);
//ascending sort
CLSPARSE_EXPORT clsparseStatus
radix_sort_by_key(
      int keys_first,
      int keys_last,
      int values_first,
      cl_mem clInputKeys,
      cl_mem clInputValues,
      cl_mem clInputValues2,
      int float_type, //todo make it a template
      clsparseControl control
);

