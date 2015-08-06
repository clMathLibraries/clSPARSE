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
#ifndef _CLSPARSE_ELEMENTWISE_OPERATORS_HPP_
#define _CLSPARSE_ELEMENTWISE_OPERATORS_HPP_

enum ElementWiseOperator
{
    EW_PLUS = 0,
    EW_MINUS,
    EW_MULTIPLY,
    EW_DIV,
    EW_DUMMY //does nothing
};


template<ElementWiseOperator OP>
struct ElementWiseOperatorTrait {};

#define EW_OP_GENERATOR(OP) \
    template<> \
    struct ElementWiseOperatorTrait<OP> { \
    static const char* operation;};

EW_OP_GENERATOR(EW_PLUS)
EW_OP_GENERATOR(EW_MINUS)
EW_OP_GENERATOR(EW_MULTIPLY)
EW_OP_GENERATOR(EW_DIV)
EW_OP_GENERATOR(EW_DUMMY)


#endif //ELEMENTWISE_HPP
