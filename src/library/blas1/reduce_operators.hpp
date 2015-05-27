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
