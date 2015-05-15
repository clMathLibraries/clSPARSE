#pragma once
#ifndef _CLSPARSE_REDUCE_OPERATORS_HPP_
#define _CLSPARSE_REDUCE_OPERATORS_HPP_

enum ReduceOperator
{
    PLUS = 0,
    SQR,
    SQRT,
    FABS,
    DUMMY //does nothing
};


template<ReduceOperator OP>
struct ReduceOperatorTrait {};

#define REDUCE_OP_GENERATOR(OP) \
    template<> \
    struct ReduceOperatorTrait<OP> { \
    static const char* operation;};

REDUCE_OP_GENERATOR(PLUS)
REDUCE_OP_GENERATOR(SQR)
REDUCE_OP_GENERATOR(SQRT)
REDUCE_OP_GENERATOR(FABS)
REDUCE_OP_GENERATOR(DUMMY)




#endif //_CLSPARSE_REDUCE_OPERATORS_HPP_
