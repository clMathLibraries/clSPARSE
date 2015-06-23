#ifndef _CLSPARSE_VALIDATE_H_
#define _CLSPARSE_VALIDATE_H_

#include "include/clSPARSE-private.hpp"
#include <stdio.h>

clsparseStatus
validateMemObject(clsparseScalarPrivate& scalar, size_t required_size);

clsparseStatus
validateMemObject(clsparseVectorPrivate& vector, size_t required_size);


clsparseStatus
validateMemObject( cl_mem mem, size_t required_size );

clsparseStatus
validateMemObject(void* mem, size_t required_size);


/*
* Validate cl_mem buffers regarding required size including offset;
* element_size - size of vector element in bytes, sizeof(T)
* count - number of elements,
* mem - object to validate
* off_mem - offset of first element of vector mem counted in elements
*/
clsparseStatus
    validateMemObjectSize( size_t element_size,
    size_t count,
    cl_mem mem,
    size_t off_mem );


#endif //_CLSPARSE_VALIDATE_H_
