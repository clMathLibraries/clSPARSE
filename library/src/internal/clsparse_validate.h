#ifndef _CLSPARSE_VALIDATE_H_
#define _CLSPARSE_VALIDATE_H_

#include "clSPARSE.h"

#ifdef __cplusplus
extern "C" {
#endif

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
    //check if mem object have valud required size
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



#ifdef __cplusplus
}
#endif

#endif //_CLSPARSE_VALIDATE_H_
