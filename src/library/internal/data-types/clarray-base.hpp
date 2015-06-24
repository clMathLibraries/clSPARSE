#pragma once
#ifndef _CLSPARSE_ARRAY_BASE_HPP_
#define _CLSPARSE_ARRAY_BASE_HPP_

#include "include/clSPARSE-private.hpp"
#include "internal/clsparse-control.hpp"

#include "blas1/elementwise-transform.hpp"

#include <cassert>

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
    BUFF_TYPE& data()
    {
        return _buff;
    }

    /**
     * @brief buffer
     * @return original underlying buffer
     */
    const BUFF_TYPE& data() const
    {
        return _buff;
    }

    size_t size()
    {
        return _size;
    }

    const size_t size() const
    {
        return _size;
    }


    void clear()
    {
        _buff = BUFF_TYPE();
        _size = 0;
    }

    //    /** removing copy operations, will be propagated to composed types*/
    //    this is just temporary not to use array in wrong way.

    array_base() = default;
    array_base(const array_base&) = delete;
    //array_base(array_base&&) = default;
    array_base& operator=(const array_base&) = delete;
    //array_base& operator=(array_base&&) = default;

protected:
    BUFF_TYPE _buff;
    size_t _size;
}; // array_base



// Maybe we can put base shared functions here
namespace internal
{

template <typename Container, typename T>
cl_int fill(Container& c, const T & value, cl::CommandQueue& queue)
{
    cl::Event controlEvent;
    cl_int status;

    assert (c.size() > 0);
    if (c.size() > 0)
    {
        status = queue.enqueueFillBuffer(c.data(), value, 0,
                                         c.size() * sizeof(T),
                                         NULL, &controlEvent);
        OPENCL_V_THROW(status, "queue.enqueueFillBuffer");

        status = controlEvent.wait();
        OPENCL_V_THROW(status, "controlEvent.wait");
    }

    return status;
}

template <typename Container, typename T>
cl_int fill(Container& c, const T & value)
{
    cl::Event controlEvent;
    cl_int status;

    assert (c.size() > 0);
    if (c.size() > 0)
    {
        status = c.getQueue().enqueueFillBuffer(c.data(), value, 0,
                                         c.size() * sizeof(T),
                                         NULL, &controlEvent);
        OPENCL_V_THROW(status, "queue.enqueueFillBuffer");

        status = controlEvent.wait();
        OPENCL_V_THROW(status, "controlEvent.wait");
    }

    return status;
}

template <typename T, ElementWiseOperator OP>
clsparseStatus elementwise_operation(array_base<T>& r,
                                     const array_base<T>& x,
                                     const array_base<T>& y,
                                     clsparseControl control)
{
    assert (x.size() == y.size());
    assert (x.size() == r.size());

    clsparseStatus status =
        elementwise_transform<T, OP>(r, x, y, control);

    return status;
}

template <typename T>
clsparseStatus add(array_base<T>& r,
                   const array_base<T>& x,
                   const array_base<T>& y,
                   clsparseControl control)
{
    return elementwise_operation<T, EW_PLUS>(r, x, y, control);
}

template <typename T>
clsparseStatus sub(array_base<T>& r,
                   const array_base<T>& x,
                   const array_base<T>& y,
                   clsparseControl control)
{
   return elementwise_operation<T, EW_MINUS>(r, x, y, control);

}

template <typename T>
clsparseStatus mul(array_base<T>& r,
                   const array_base<T>& x,
                   const array_base<T>& y,
                   clsparseControl control)
{
   return elementwise_operation<T, EW_MULTIPLY>(r, x, y, control);

}

template <typename T>
clsparseStatus div(array_base<T>& r,
                   const array_base<T>& x,
                   const array_base<T>& y,
                   clsparseControl control)
{
   return elementwise_operation<T, EW_DIV>(r, x, y, control);
}




} //namespace internal

} // namespace clsparse

#endif //_CLSPARSE_ARRAY_BASE_HPP_
