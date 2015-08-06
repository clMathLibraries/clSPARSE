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

#include "reduce-operators.hpp"

const char* ReduceOperatorTrait<RO_PLUS>::operation = "OP_RO_PLUS";
const char* ReduceOperatorTrait<RO_SQR>::operation = "OP_RO_SQR";
const char* ReduceOperatorTrait<RO_SQRT>::operation = "OP_RO_SQRT";
const char* ReduceOperatorTrait<RO_FABS>::operation = "OP_RO_FABS";
const char* ReduceOperatorTrait<RO_DUMMY>::operation = "OP_RO_DUMMY";
