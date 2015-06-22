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
