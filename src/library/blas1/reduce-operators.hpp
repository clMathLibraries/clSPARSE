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
#ifndef _CLSPARSE_REDUCE_OPERATORS_HPP_
#define _CLSPARSE_REDUCE_OPERATORS_HPP_

enum ReduceOperator
{
    RO_PLUS = 0,
    RO_SQR,
    RO_SQRT,
    RO_FABS,
    RO_DUMMY //does nothing
};


template<ReduceOperator OP>
struct ReduceOperatorTrait {};

#define REDUCE_OP_GENERATOR(OP) \
    template<> \
    struct ReduceOperatorTrait<OP> { \
    static const char* operation;};

REDUCE_OP_GENERATOR(RO_PLUS)
REDUCE_OP_GENERATOR(RO_SQR)
REDUCE_OP_GENERATOR(RO_SQRT)
REDUCE_OP_GENERATOR(RO_FABS)
REDUCE_OP_GENERATOR(RO_DUMMY)




#endif //_CLSPARSE_REDUCE_OPERATORS_HPP_
