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
#ifndef _CLSPARSE_CONVERSION_UTILS_HPP_
#define _CLSPARSE_CONVERSION_UTILS_HPP_

#include "internal/data-types/clvector.hpp"
#include "scan.hpp"
#include "reduce-by-key.hpp"
#include "blas1/reduce.hpp"

template <typename T> //index type
clsparseStatus
indices_to_offsets(clsparse::vector<T>& offsets,
                   const clsparse::vector<T>& indices,
                   const clsparseControl control)
{
    typedef typename clsparse::vector<T> IndicesArray;
    //typedef typename clsparse::vector<T>::size_type SizeType;
    typedef cl_ulong SizeType;

    //if (std::is_integral<T>)

    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    SizeType size = indices.size() > offsets.size() ? indices.size() : offsets.size();

    IndicesArray values (control, indices.size(), 1, CL_MEM_READ_WRITE, true);
    IndicesArray keys_output (control, indices.size(), 0, CL_MEM_READ_WRITE, false);
    IndicesArray values_output (control, size, 0, CL_MEM_READ_WRITE, false);

    clsparseStatus status =
            internal::reduce_by_key(keys_output, values_output,
                                    indices, values, control);

    CLSPARSE_V(status, "Error: reduce_by_key");

    if (status != clsparseSuccess)
        return status;

    assert(values_output.size() >= offsets.size());

    cl_event clEvent;
    cl_int cl_status  = clEnqueueCopyBuffer(control->queue(),
                                            values_output.data()(),
                                            offsets.data()(),
                                            0,
                                            0,
                                            offsets.size() * sizeof(T),
                                            0,
                                            nullptr,
                                            &clEvent);

    CLSPARSE_V(cl_status, "Error: Enqueue copy buffer values to offsets");

    cl_status = clWaitForEvents(1, &clEvent);

    CLSPARSE_V(cl_status, "Error: clWaitForEvents");

    cl_status = clReleaseEvent(clEvent);

    CLSPARSE_V(cl_status, "Error: clReleaseEvent");

    // Dunno why but this throws CL_INVALID_CONTEXT erro;
    //    cl::Event event;
    //    cl::enqueueCopyBuffer(values_output.data(), offsets.data(),
    //                                          0, 0, offsets.size(),
    //                                          nullptr, &event);

    //    CLSPARSE_V(cl_status, "Error: enqueueCopyBuffer");
    //    CLSPARSE_V(event.wait(), "Error: event wait");

    status = exclusive_scan<EW_PLUS>(offsets, offsets, control);

    return status;
}

template <typename T>
clsparseStatus
offsets_to_indices(clsparse::vector<T>& indices,
                   const clsparse::vector<T>& offsets,
                   typename clsparse::vector<T>::size_type num_rows,
                   const clsparseControl control)
{
    //typedef typename clsparse::vector<T>::size_type SizeType;
    typedef cl_ulong SizeType;

    assert (num_rows + 1 == offsets.size());


    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    SizeType wave_size = control->wavefront_size;
    SizeType subwave_size = wave_size;
    SizeType group_size = 256;

    SizeType size = indices.size();
    SizeType elements_per_row = size / num_rows; // assumed number elements per row;

    // adjust subwave_size according to elements_per_row;
    // each wavefront will be assigned to process to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (elements_per_row < 64) {  subwave_size = 32;  }
    }
    if (elements_per_row < 32) {  subwave_size = 16;  }
    if (elements_per_row < 16) {  subwave_size = 8;  }
    if (elements_per_row < 8)  {  subwave_size = 4;  }
    if (elements_per_row < 4)  {  subwave_size = 2;  }


    const std::string params = std::string ()
            + " -DINDEX_TYPE=" + OclTypeTraits<T>::type
            //not used in this kernel but required by program conversion_utils
            + " -DVALUE_TYPE=" + OclTypeTraits<T>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<SizeType>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    cl::Kernel kernel = KernelCache::get(control->queue, "conversion_utils",
                                         "offsets_to_indices", params);

    KernelWrap kWrapper (kernel);
    kWrapper << num_rows
             << offsets.data()
             << indices.data();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * num_rows;

    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);

    cl::NDRange local(group_size);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int cl_status = kWrapper.run(control, global, local);

    CLSPARSE_V(cl_status, "Error: offsets_to_indices_vector");

    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;

    return clsparseSuccess;
}

// V - matrix value type, I - matrix index type
template<typename V, typename I>
clsparseStatus
transform_csr_2_dense(/*csr matrix*/
                      const clsparse::vector<I>& row_offsets,
                      const clsparse::vector<I>& col_indices,
                      const clsparse::vector<V>& values,
                      typename clsparse::vector<I>::size_type num_rows,
                      typename clsparse::vector<I>::size_type num_cols,
                      /* dense matrix */
                      clsparse::vector<V>& A,
                      const clsparseControl control)
{
    //typedef typename clsparse::vector<I>::size_type SizeType;
    typedef cl_ulong SizeType;

    if (!clsparseInitialized)
    {
        return clsparseNotInitialized;
    }

    //check opencl elements
    if (control == nullptr)
    {
        return clsparseInvalidControlObject;
    }

    SizeType wave_size = control->wavefront_size;
    SizeType subwave_size = wave_size;
    SizeType group_size = 256;

    SizeType elements_per_row = values.size() / num_rows; // assumed number elements per row;

    // adjust subwave_size according to elements_per_row;
    // each wavefront will be assigned to process to the row of the csr matrix
    if(wave_size > 32)
    {
        //this apply only for devices with wavefront > 32 like AMD(64)
        if (elements_per_row < 64) {  subwave_size = 32;  }
    }
    if (elements_per_row < 32) {  subwave_size = 16;  }
    if (elements_per_row < 16) {  subwave_size = 8;  }
    if (elements_per_row < 8)  {  subwave_size = 4;  }
    if (elements_per_row < 4)  {  subwave_size = 2;  }


    const std::string params = std::string ()
            + " -DVALUE_TYPE=" + OclTypeTraits<V>::type
            + " -DINDEX_TYPE=" + OclTypeTraits<I>::type
            + " -DSIZE_TYPE=" + OclTypeTraits<SizeType>::type
            + " -DWG_SIZE=" + std::to_string(group_size)
            + " -DWAVE_SIZE=" + std::to_string(wave_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(subwave_size);

    cl::Kernel kernel = KernelCache::get(control->queue, "conversion_utils",
                                         "transform_csr_to_dense", params);


    KernelWrap kWrapper (kernel);
    kWrapper << num_rows
             << num_cols
             << row_offsets.data()
             << col_indices.data()
             << values.data()
             << A.data();

    // subwave takes care of each row in matrix;
    // predicted number of subwaves to be executed;
    cl_uint predicted = subwave_size * num_rows;

    cl_uint global_work_size =
            group_size* ((predicted + group_size - 1 ) / group_size);

    cl::NDRange local(group_size);
    cl::NDRange global(global_work_size > local[0] ? global_work_size : local[0]);

    cl_int cl_status = kWrapper.run(control, global, local);

    CLSPARSE_V(cl_status, "Error: csr 2 dense");

    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;

    return clsparseSuccess;

}

/* *
 * Function calculates the position of nnz in matrix A
 * returns:
 *  nnz_locations filled vector for which the indexes of values == 1 points to position
 *  num_nonzeros number of nonzeros in A
 */
template <typename V, typename I>
clsparseStatus
calculate_num_nonzeros(/*dense matrix*/
                       const clsparse::vector<V>& A,
                       clsparse::vector<I>& nnz_locations,
                       I& num_nonzeros,
                       const clsparseControl control)
{
    typedef cl_ulong SizeType;

    SizeType dense_size = A.size();

    SizeType workgroup_size   = 256;
    SizeType global_work_size = 0;

    if (dense_size % workgroup_size == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / workgroup_size * workgroup_size + workgroup_size;

    if (dense_size < workgroup_size) global_work_size = workgroup_size;

    const std::string params = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<I>::type
            + " -DSIZE_TYPE="  + OclTypeTraits<SizeType>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<V>::type
            + " -DWG_SIZE=" + std::to_string(workgroup_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(2); //required by program;

    //cl::Kernel kernel = KernelCache::get(control->queue,"dense2csr", "process_scaninput", params);
    cl::Kernel kernel = KernelCache::get(control->queue,"conversion_utils", "scan_nonzero_locations", params);

    KernelWrap kWrapper(kernel);

    kWrapper << dense_size
             << A.data()
             << nnz_locations.data();

    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_work_size);

    cl_int cl_status = kWrapper.run(control, global, local);

    CLSPARSE_V(cl_status, "Error process scaninput");

    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;

    //TODO: is it just write_only?
    clsparse::vector<I> nnz (control, 1, 0, CL_MEM_READ_WRITE, false);

    //due to this definition nnz and nnz_location have to be of the same type;
    clsparseStatus status = reduce<I, RO_PLUS>(nnz, nnz_locations, control);

    CLSPARSE_V(status, "Error: reduce");

    if (status!= clsparseSuccess)
        return clsparseInvalidKernelExecution;

    num_nonzeros = nnz[0];
    //std::cout << "NNZ: " << num_nonzeros << std::endl;
    return status;
}

/* *
 * This function converts the dense matrix to coo matrix type.
 * coo matrix is filled and allocated here.
 */

template<typename V, typename I>
clsparseStatus
dense_to_coo(clsparseCooMatrix* coo,
             const clsparse::vector<V>& A,
             const clsparse::vector<I>& nnz_locations,
             const clsparse::vector<I>& coo_indexes,
             const clsparseControl control)
{
    typedef cl_ulong SizeType;

    assert(coo->num_nonzeros > 0);
    assert(coo->num_cols > 0);
    assert(coo->num_rows > 0);

    assert(A.size() > 0);
    assert(nnz_locations.size() > 0);
    assert(coo_indexes.size() > 0);

    SizeType dense_size = A.size();

    cl_int cl_status;

    coo->values = clCreateBuffer( control->getContext()(), CL_MEM_READ_WRITE,
                                  coo->num_nonzeros * sizeof(V), NULL, &cl_status );
    CLSPARSE_V(cl_status, "Create coo values buffer");

    coo->colIndices = clCreateBuffer( control->getContext()(), CL_MEM_READ_WRITE,
                                      coo->num_nonzeros * sizeof(I), NULL, &cl_status );
    CLSPARSE_V(cl_status, "Create coo col indices buffer");

    coo->rowIndices = clCreateBuffer(control->getContext()(), CL_MEM_READ_WRITE,
                                     coo->num_nonzeros * sizeof(I), NULL, &cl_status );
    CLSPARSE_V(cl_status, "Create coo row indices buffer");



    SizeType workgroup_size   = 256;
    SizeType global_work_size = 0;

    if (dense_size % workgroup_size == 0)
        global_work_size = dense_size;
    else
        global_work_size = dense_size / workgroup_size * workgroup_size + workgroup_size;

    if (dense_size < workgroup_size) global_work_size = workgroup_size;

    const std::string params = std::string()
            + " -DINDEX_TYPE=" + OclTypeTraits<I>::type
            + " -DSIZE_TYPE="  + OclTypeTraits<SizeType>::type
            + " -DVALUE_TYPE=" + OclTypeTraits<V>::type
            + " -DWG_SIZE=" + std::to_string(workgroup_size)
            + " -DSUBWAVE_SIZE=" + std::to_string(2); //required by program;

    //cl::Kernel kernel = KernelCache::get(control->queue,"dense2csr", "spread_value", params);
    cl::Kernel kernel = KernelCache::get(control->queue,"conversion_utils",
                                         "scatter_coo_locations", params);

    KernelWrap kWrapper(kernel);

    kWrapper << (SizeType) coo->num_rows
             << (SizeType) coo->num_cols
             << (SizeType) dense_size
             << A.data()
             << nnz_locations.data()
             << coo_indexes.data()
             << coo->rowIndices
             << coo->colIndices
             << coo->values;

    cl::NDRange local(workgroup_size);
    cl::NDRange global(global_work_size);

    cl_status = kWrapper.run(control, global, local);

    CLSPARSE_V(cl_status, "Error process scaninput");

    if (cl_status != CL_SUCCESS)
        return clsparseInvalidKernelExecution;

    return clsparseSuccess;
}

#endif //_CLSPARSE_CONVERSION_UTILS_HPP_
