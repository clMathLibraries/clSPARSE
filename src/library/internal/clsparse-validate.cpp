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

#include "clsparse-validate.hpp"
#include "ocl-type-traits.hpp"
#include <iostream>

clsparseStatus
validateMemObject(clsparseScalarPrivate &scalar, size_t required_size)
{
#if (BUILD_CLVERSION >= 200)
    std::cout << "Don't know how to validate SVM void* buffer" << std::endl;
    return clsparseSuccess;
#else
    return validateMemObject(scalar.value, required_size);
#endif
}

clsparseStatus
validateMemObject(cldenseVector &vector, size_t required_size)
{
#if (BUILD_CLVERSION >= 200)
    std::cout << "Don't know how to validate SVM void* buffer" << std::endl;
    return clsparseSuccess;
#else
    return validateMemObject(vector.values, required_size);
#endif
}

clsparseStatus
validateMemObject(void* mem, size_t required_size)
{
    std::cout << "validateMemObject void* buffer" << std::endl;

    return clsparseSuccess;
}

clsparseStatus
validateMemObject( cl_mem mem, size_t required_size)
{
    //check if valid mem object,
    cl_mem_object_type mem_type = 0;
    clGetMemObjectInfo(mem, CL_MEM_TYPE, sizeof(mem_type), &mem_type, NULL);
    if(mem_type != CL_MEM_OBJECT_BUFFER)
    {
        return clsparseInvalidMemObj;
    }
    //check if mem object have valid required size
    if (required_size > 0)
    {
        size_t current_size;
        clGetMemObjectInfo(mem, CL_MEM_SIZE,
                            sizeof(current_size), &current_size, NULL);
        if(current_size != required_size)
            return clsparseInvalidSize;
#ifndef NDEBUG
     printf("Mem size: %lu\n", current_size);
     printf("Required size: %lu\n", required_size);
#endif
    }

    return clsparseSuccess;
}

/*
 * Validate cl_mem buffers regarding required size including offset;
 * element_size - size of vector element in bytes, sizeof(T)
 * count - number of elements,
 * mem - object to validate
 * off_mem - offset of first element of vector mem counted in elements
 */
clsparseStatus
validateMemObjectSize(size_t element_size,
                       size_t count,
                       cl_mem mem,
                       size_t off_mem)
{
    size_t mem_size; //cl_mem current size
    size_t vec_size = count * element_size;
    off_mem *= element_size; //it's a copy

    if (count == 0)
    {
        return clsparseInvalidSize;
    }

    cl_int status =
            clGetMemObjectInfo(mem, CL_MEM_SIZE,
                               sizeof(mem_size), &mem_size, NULL);
    if (status != CL_SUCCESS)
    {
        return clsparseInvalidMemObj;
    }

    if ((off_mem + vec_size > mem_size) || (off_mem + vec_size < off_mem))
    {
        return clsparseInsufficientMemory;
    }

    return clsparseSuccess;
}
