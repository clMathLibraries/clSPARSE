#pragma once
#ifndef _CLSPARSE_CLARRAY_HPP_
#define _CLSPARSE_CLARRAY_HPP_

#include "internal/clsparse_control.hpp"
#include "include/clSPARSE-private.hpp"
#include "clsparse.error.hpp"

#include "clarray_base.hpp"
#include "reference_base.hpp"

#include <cassert>

/* First approach to implement clsparse::array type for internal use
 * Container of array is cl::Buffer it is easier to use
 */
namespace clsparse
{

template <typename T> class array;

template <typename T>
class array : public array_base<T>
{

    typedef array_base<T> BASE;

public:

    typedef typename array_base<T>::value_type value_type;

    typedef reference_base<array<value_type> > reference;

    template <typename Container>
    class iterator_base
    {
    public:

        typedef iterator_base self_type;

        typedef typename Container::value_type value_type;

        typedef std::forward_iterator_tag iterator_category;

        typedef size_t difference_type;

        iterator_base(Container& rhs, difference_type index, cl::CommandQueue queue):
            container ( rhs ), index(index), range(1), queue(queue)
        {}

        iterator_base(Container& rhs, difference_type index, difference_type range, cl::CommandQueue queue):
            container ( rhs ), index ( index ), range ( range ), queue ( queue )
        {}

        iterator_base( const iterator_base& iter) :
            container (iter.container), index (iter.index), range (iter.range), queue(iter.queue)
        {

        }

        const reference operator*()
        {
            return reference( container, index, queue);
        }

        iterator_base < Container >& operator+= ( const difference_type& n)
        {
            index += n;
            return *this;
        }

        iterator_base < Container >& operator = ( const difference_type& n)
        {
            index += n;
            return *this;
        }

        const iterator_base < Container > operator+ ( const difference_type& n) const
        {
            iterator_base < Container > result(*this);
            result.index += n;
            return result;
        }

        bool operator== (const self_type& rhs ) const
        {
            bool sameIndex = rhs.index == index;
            bool sameContainer = (&rhs.container == &container);
            return (sameContainer && sameIndex );
        }

        bool operator != (const self_type& rhs) const
        {
            bool sameIndex = rhs.index == index;
            bool sameContainer = (&rhs.container == &container);
            return !(sameContainer && sameIndex );
        }

    private:

        Container& container;
        difference_type index;
        difference_type range;
        cl::CommandQueue queue;
    };

    typedef iterator_base< array<value_type> > iterator;

    array(clsparseControl control, size_t size, const value_type& value = value_type(),
          cl_mem_flags flags = CL_MEM_READ_WRITE, cl_bool init = true) : queue(control->queue)
    {
        BASE::buff = create_buffer(size, flags);

        if (init)
        {
            cl_int status = fill(control, value);
            OPENCL_V_THROW(status, "array.fill");
        }
    }

    //create array from preinitialized buffer.
    array (clsparseControl control, const cl::Buffer& buffer, size_t size)
        : BASE::buff(buffer), _size(size), queue(control->queue)
    {}

    array (clsparseControl control, const cl_mem& mem, size_t size)
        : _size(size), queue(control->queue)
    {

        //operator = on Memory class transfers ownership which I want to avoid;
         clRetainMemObject(mem);
         BASE::buff = mem;
    }


    array(const array& other) : _size(other._size), queue(other.queue)
    {
        cl_int status;
        cl::Event controlEvent;

        const cl::Buffer& src = other.buffer();
        cl::Buffer& dst = BASE::buffer();

        cl_mem_flags flags = src.getInfo<CL_MEM_FLAGS>(&status);

        OPENCL_V_THROW(status, "Array cpy constr, getInfo<CL_MEM_FLAGS>");

        assert (_size > 0);

        dst = create_buffer(_size, flags);

        status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                         sizeof(value_type) * other.size(),
                                         NULL, &controlEvent);
        OPENCL_V_THROW(status, "operator= queue.enqueueCopyBuffer");
        status = controlEvent.wait();
        OPENCL_V_THROW(status, "operator= controlEvent.wait");

    }


    //returns deep copy of the object
    array copy(clsparseControl control)
    {
        cl::Event controlEvent;
        cl_int status;

        assert(_size > 0);

        if (_size > 0)
        {
            const cl::Buffer& src = BASE::buff;


            array new_array(control, _size);
            cl::Buffer& dst = new_array.buffer();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * _size,
                                             NULL, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            OPENCL_V_THROW(status, "controlEvent.wait");
            return new_array;
        }

    }

    cl_int fill(clsparseControl control, const T &value)
    {
        cl::Event controlEvent;
        cl_int status;

        assert (_size > 0);
        if (_size > 0)
        {
            status = queue.enqueueFillBuffer(BASE::buff, value, 0,
                                             _size * sizeof(value_type),
                                             NULL, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueFillBuffer");

            status = controlEvent.wait();
            OPENCL_V_THROW(status, "controlEvent.wait");
        }

        return status;
    }

    size_t size() const
    {
        return _size;
    }

    void resize(size_t size)
    {
        if(this->_size != size)
        {
            BASE::buff = create_buffer(size);

        }
    }

    array shallow_copy()
    {
        array tmpCopy;
        tmpCopy._size = _size;
        tmpCopy.buff = this->buff;
        return tmpCopy;
    }

    void clear()
    {
        BASE::clear();
        _size = 0;
    }


    reference operator[]( size_t n )
    {
        assert(n < _size);

        return reference( *this, n, queue);
    }


    iterator begin()
    {
        return iterator(*this, 0, queue);
    }

    iterator end()
    {
        return iterator(*this, _size, queue);
    }


    //assignment operator performs deep copy
    array& operator= (const array& other)
    {
        if (this != &other)
        {
            assert(other.size() > 0);

            if (_size != other.size())
                resize(other.size());

            cl::Event controlEvent;
            cl_int status;

            const cl::Buffer& src = other.buffer();
            cl::Buffer& dst = BASE::buffer();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * other.size(),
                                             NULL, &controlEvent);
            OPENCL_V_THROW(status, "operator= queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            OPENCL_V_THROW(status, "operator= controlEvent.wait");
        }
        return *this;

    }


private:

    cl::Buffer create_buffer(size_t size, cl_map_flags flags = CL_MEM_READ_WRITE)
    {
        if(size > 0)
        {
            _size = size;
            cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
            return ::cl::Buffer(ctx, flags, sizeof(value_type) * _size);
        }
    }

    size_t _size;
    cl::CommandQueue queue;

}; // array

} //namespace clsparse


#endif //_CLSPARSE_CLARRAY_HPP_
