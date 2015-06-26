#pragma once
#ifndef _CLSPARSE_CLARRAY_HPP_
#define _CLSPARSE_CLARRAY_HPP_

#include "internal/clsparse-control.hpp"
#include "include/clSPARSE-private.hpp"
#include "clSPARSE-error.h"

#include "clarray-base.hpp"
#include "reference-base.hpp"
#include "iterator-base.hpp"

#include "blas1/elementwise-transform.hpp"

#include <cassert>

#include <array>

/**
 *  @brief Device container for storing a fixed size sequence of elements
 *
 *  @tparam T Type of element
 *  @tparam N Number of elements
 */
namespace clsparse
{
    template <typename T, size_t N> class array;

template <typename T, size_t N>
class array : public array_base<T>
{
    typedef array_base<T> BASE;
    typedef typename BASE::BUFF_TYPE BUFF_TYPE;

public:

    typedef typename BASE::value_type value_type;

    typedef array<value_type, N> self_type;

    typedef reference_base<self_type> reference;

    typedef iterator_base<self_type> iterator;

    /** I know that std::array do not have any constructors but we to pass info
     * about the buffer in nice way. I think no other is required
     * @brief array
     * @param control
     * @param value
     * @param flags
     * @param init
     */
    array(clsparseControl control, const value_type& value = value_type(),
          cl_mem_flags flags = CL_MEM_READ_WRITE, cl_bool init = true) :
        queue(control->queue)
    {
        BASE::data() = create_buffer(N, flags);
        if (init)
        {
            cl_int status = fill(control, value);
            CLSPARSE_V(status, "array.fill");
        }
    }

    array (const self_type& other, bool copy = true) : queue(other.queue)
    {
        cl_int status;
        cl::Event controlEvent;

        const BUFF_TYPE& src = other.data();
        BUFF_TYPE& dst = BASE::data();

        cl_mem_flags flags;
        status = src.getInfo(CL_MEM_FLAGS, &flags);
        CLSPARSE_V(status, "Array cpy constr, getInfo<CL_MEM_FLAGS>");

        assert (BASE::_size > 0);

        dst = create_buffer(BASE::_size, flags);

        if (copy)
        {
            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                         sizeof(value_type) * other.size(),
                                         NULL, &controlEvent);
            CLSPARSE_V(status, "operator= queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            CLSPARSE_V(status, "operator= controlEvent.wait");
        }
    }

    array& operator=(const self_type& other)
    {
        if (this != &other)
        {
            assert(other.size() > 0);

            cl::Event controlEvent;
            cl_int status;

            const cl::Buffer& src = other.data();
            cl::Buffer& dst = BASE::data();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * other.size(),
                                             NULL, &controlEvent);
            CLSPARSE_V(status, "operator= queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            CLSPARSE_V(status, "operator= controlEvent.wait");
        }
        return *this;

    }

    size_t size() const
    {
        return BASE::_size;
    }

    cl_int fill(clsparseControl control, const T &value)
    {
        return internal::fill(*this, value);
    }

//    void swap (array<T, N>& other)
//    {
//        if (this == &other)
//            return;

//        std::swap(*this, other);
//    }

    reference operator[] ( size_t n)
    {
        assert (n < size());
        return reference( *this, n, queue);
    }

    reference at (size_t n )
    {
        assert (n < size());
        return reference( *this, n, queue);
    }

    size_t max_size ()
    {
        return size();
    }

    bool empty()
    {
        return N == 0 ? true : false;
    }

    iterator begin()
    {
        return iterator(*this, 0, queue);
    }


    iterator end()
    {
        return iterator(*this, size(), queue);
    }

    //proper signature should be following but we need const_iterator
    //bool operator ==(const self_type& rhs) const
    bool operator ==(self_type& rhs)
    {
        //very very nasty; should be done with kernel;
        bool sameSize = size() == rhs.size();
        bool sameContent = false;

        if (sameSize)
        for (auto tIt = this->begin(), oIt = rhs.begin(); tIt != this->end(); ++tIt, ++oIt)
        {
            sameContent = *tIt == *oIt;
        }

        return (sameSize && sameContent);
    }

    //same as above
    bool operator !=(self_type& rhs)
    {
        return this->operator ==(rhs);
    }


    clsparseStatus add(const self_type& x, const self_type& y, clsparseControl control)
    {
        return internal::add(*this, x, y, control);
    }


    clsparseStatus add(const self_type& x, clsparseControl control)
    {
        return add(*this, x, control);
    }

    clsparseStatus sub(const self_type& x, const self_type& y, clsparseControl control)
    {
        return internal::sub(*this, x, y, control);
    }

    clsparseStatus sub(const self_type& x, clsparseControl control)
    {
        return sub(*this, x, control);
    }

    clsparseStatus mul(const self_type& x, const self_type& y, clsparseControl control)
    {
        return internal::mul(*this, x, y, control);
    }

    clsparseStatus mul(const self_type& x, clsparseControl control)
    {
        return mul(*this, x, control);
    }

    clsparseStatus div(const self_type& x, const self_type& y, clsparseControl control)
    {
        return internal::div(*this, x, y, control);
    }

    clsparseStatus div(const self_type& x, clsparseControl control)
    {
        return div(*this, x, control);
    }

    cl::CommandQueue& getQueue()
    {
        return queue;
    }

private:

    cl::Buffer create_buffer(size_t size, cl_map_flags flags = CL_MEM_READ_WRITE)
    {
        if(size > 0)
        {
            BASE::_size = size;
            cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
            return ::cl::Buffer(ctx, flags, sizeof(value_type) * BASE::_size);
        }

		return cl::Buffer();
    }
private:
  cl::CommandQueue& queue;
};

template<typename T> using scalar = array<T, 1>;
} // namespace clsparse

#endif //_CLSPARSE_CLARRAY_HPP_
