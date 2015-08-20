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

#ifndef _CLSPARSE_VALIDATE_H_
#define _CLSPARSE_VALIDATE_H_

#include "include/clSPARSE-private.hpp"
#include <stdio.h>

clsparseStatus
validateMemObject(clsparseScalarPrivate& scalar, size_t required_size);

clsparseStatus
validateMemObject(cldenseVectorPrivate& vector, size_t required_size);


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
