#include "reduce_operators.hpp"

const char* ReduceOperatorTrait<PLUS>::operation = "OP_PLUS";
const char* ReduceOperatorTrait<SQR>::operation = "OP_SQR";
const char* ReduceOperatorTrait<SQRT>::operation = "OP_SQRT";
const char* ReduceOperatorTrait<FABS>::operation = "OP_FABS";
const char* ReduceOperatorTrait<DUMMY>::operation = "OP_DUMMY";

