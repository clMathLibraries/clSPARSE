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
