#include "clsparse_validate.h"

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
