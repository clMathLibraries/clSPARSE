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

#ifndef _IO_EXCEPTION_HPP_
#define _IO_EXCEPTION_HPP_

#include <stdexcept>

namespace clsparse
{

// Input output exception for indicating particular problems
// related to reading files from disk
class io_exception : public std::runtime_error
{
public:


    explicit
    io_exception(const std::string& arg) : std::runtime_error(arg)
    {}



};

} // namespace clsparse


#endif //_IO_EXCEPTION_HPP_
