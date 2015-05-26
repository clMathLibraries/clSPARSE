#pragma once
#ifndef _CLSPARSE_ARRAY_BASE_HPP_
#define _CLSPARSE_ARRAY_BASE_HPP_

#include "include/clSPARSE-private.hpp"

namespace clsparse
{

//template <typename T>
//struct clContainer;


struct clContainer
{
    typedef cl::Buffer type;
};



template <typename T>
class array_base
{
public:

    typedef T value_type;
    //typedef typename clContainer<T>::type BUFF_TYPE;
    typedef typename clContainer::type BUFF_TYPE;

    /**
     * @brief buffer
     * @return original underlying buffer
     */
    BUFF_TYPE& buffer()
    {
        return buff;
    }

    /**
     * @brief buffer
     * @return original underlying buffer
     */
    const BUFF_TYPE& buffer() const
    {
        return buff;
    }

    BUFF_TYPE& operator()()
    {
        return buff;
    }

    const BUFF_TYPE& operator()() const
    {
        return buff;
    }

    void clear()
    {
        buff = BUFF_TYPE();
    }

    //    /** removing copy operations, will be propagated to composed types*/
    //    this is just temporary not to use array in wrong way.

    array_base() = default;
    array_base(const array_base&) = delete;
    array_base(array_base&&) = default;
    array_base& operator=(const array_base&) = delete;
    array_base& operator=(array_base&&) = default;

protected:
    BUFF_TYPE buff;
}; // array_base

} // namespace clsparse

#endif //_CLSPARSE_ARRAY_BASE_HPP_
