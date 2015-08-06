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

