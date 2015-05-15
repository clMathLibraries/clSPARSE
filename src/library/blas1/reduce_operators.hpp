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

#define SCAN_OP_GENERATOR(OP) \
    template<> \
    struct ReduceOperatorTrait<OP> { \
    static const char* operation;};

SCAN_OP_GENERATOR(PLUS)
SCAN_OP_GENERATOR(SQR)
SCAN_OP_GENERATOR(SQRT)
SCAN_OP_GENERATOR(FABS)
SCAN_OP_GENERATOR(DUMMY)




#endif //_CLSPARSE_REDUCE_OPERATORS_HPP_
