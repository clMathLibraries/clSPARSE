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

#include "elementwise-operators.hpp"

const char* ElementWiseOperatorTrait<EW_PLUS>::operation = "OP_EW_PLUS";
const char* ElementWiseOperatorTrait<EW_MINUS>::operation = "OP_EW_MINUS";
const char* ElementWiseOperatorTrait<EW_MULTIPLY>::operation = "OP_EW_MULTIPLY";
const char* ElementWiseOperatorTrait<EW_DIV>::operation = "OP_EW_DIV";
const char* ElementWiseOperatorTrait<EW_MIN>::operation = "OP_EW_MIN";
const char* ElementWiseOperatorTrait<EW_MAX>::operation = "OP_EW_MAX";
const char* ElementWiseOperatorTrait<EW_DUMMY>::operation = "OP_EW_DUMMY";
