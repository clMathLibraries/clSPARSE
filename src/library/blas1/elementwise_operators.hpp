#pragma once
#ifndef _CLSPARSE_ELEMENTWISE_OPERATORS_HPP_
#define _CLSPARSE_ELEMENTWISE_OPERATORS_HPP_

enum ElementWiseOperator
{
    PLUS = 0,
    MINUS,
    MULTIPLY,
    DUMMY //does nothing
};


template<ElementWiseOperator OP>
struct ElementWiseOperatorTrait {};

#define EW_OP_GENERATOR(OP) \
    template<> \
    struct ElementWiseOperatorTrait<OP> { \
    static const char* operation;};

EW_OP_GENERATOR(PLUS)
EW_OP_GENERATOR(MINUS)
EW_OP_GENERATOR(MULTIPLY)
EW_OP_GENERATOR(DUMMY)

#endif //ELEMENTWISE_HPP
