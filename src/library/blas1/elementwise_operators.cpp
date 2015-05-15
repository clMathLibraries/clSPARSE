#include "elementwise_operators.hpp"

const char* ElementWiseOperatorTrait<PLUS>::operation = "OP_PLUS";
const char* ElementWiseOperatorTrait<MINUS>::operation = "OP_MINUS";
const char* ElementWiseOperatorTrait<MULTIPLY>::operation = "OP_MULTIPLY";
const char* ElementWiseOperatorTrait<DUMMY>::operation = "OP_DUMMY";
