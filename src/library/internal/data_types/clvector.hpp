#pragma once
#ifndef _CLSPARSE_CLARRAY_HPP_
#define _CLSPARSE_CLARRAY_HPP_

#include "internal/clsparse_control.hpp"
#include "include/clSPARSE-private.hpp"
#include "clsparse.error.hpp"

#include "clarray_base.hpp"
#include "reference_base.hpp"

#include "blas1/elementwise_transform.hpp"

#include <cassert>

/* First approach to implement clsparse::array type for internal use
 * Container of array is cl::Buffer it is easier to use
 */
namespace clsparse
{

template <typename T> class vector;

template <typename T>
class vector : public array_base<T>
{

    typedef array_base<T> BASE;

public:

    typedef typename array_base<T>::value_type value_type;

    typedef reference_base<vector<value_type> > reference;

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

    typedef iterator_base< vector<value_type> > iterator;

    vector(clsparseControl control, size_t size, const value_type& value = value_type(),
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
    vector (clsparseControl control, const cl::Buffer& buffer, size_t size)
        : BASE::buff(buffer), _size(size), queue(control->queue)
    {}

    vector (clsparseControl control, const cl_mem& mem, size_t size)
        : _size(size), queue(control->queue)
    {

        //operator = on Memory class transfers ownership which I want to avoid;
         clRetainMemObject(mem);
         BASE::buff = mem;
    }


    vector(const vector& other, bool copy = true) : _size(other._size), queue(other.queue)
    {
        cl_int status;
        cl::Event controlEvent;

        const cl::Buffer& src = other.data();
        cl::Buffer& dst = BASE::data();

        cl_mem_flags flags = src.getInfo<CL_MEM_FLAGS>(&status);

        OPENCL_V_THROW(status, "Array cpy constr, getInfo<CL_MEM_FLAGS>");

        assert (_size > 0);

        dst = create_buffer(_size, flags);

        if (copy)
        {
            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                         sizeof(value_type) * other.size(),
                                         NULL, &controlEvent);
            OPENCL_V_THROW(status, "operator= queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            OPENCL_V_THROW(status, "operator= controlEvent.wait");
        }

    }


    //returns deep copy of the object
    vector copy(clsparseControl control)
    {
        cl::Event controlEvent;
        cl_int status;

        assert(_size > 0);

        if (_size > 0)
        {
            const cl::Buffer& src = BASE::buff;


            vector new_vector(control, _size);
            cl::Buffer& dst = new_vector.data();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * _size,
                                             NULL, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            OPENCL_V_THROW(status, "controlEvent.wait");
            return new_vector;
        }

    }

    cl_int fill(clsparseControl control, const T &value)
    {
        cl::Event controlEvent;
        cl_int status;

        assert (_size > 0);
        if (_size > 0)
        {
            status = queue.enqueueFillBuffer(BASE::data(), value, 0,
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

    vector shallow_copy()
    {
        vector tmpCopy;
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
    vector& operator= (const vector& other)
    {
        if (this != &other)
        {
            assert(other.size() > 0);

            if (_size != other.size())
                resize(other.size());

            cl::Event controlEvent;
            cl_int status;

            const cl::Buffer& src = other.data();
            cl::Buffer& dst = BASE::data();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * other.size(),
                                             NULL, &controlEvent);
            OPENCL_V_THROW(status, "operator= queue.enqueueCopyBuffer");
            status = controlEvent.wait();
            OPENCL_V_THROW(status, "operator= controlEvent.wait");
        }
        return *this;

    }


    /* Interface for 'operator op' require zero or one arguments, this will result
     * in need of creation a new buffer which I want to avoid. Other issue,
     * is related to passing clsparseContol object to properly call the kernel.
     * It have to be passed to KernelWrapper.run function. With operator implementation
     * i can't do that nicely.
     *
     * My proposal is to implement operators as explicit functions which updates
     * the this->data() with provided parameters;
     *
     * '/' == div(const vector&x, const vector& y) { this->data = x.data() / y.data();
     * etc.
     */

    // z = x + y
    clsparseStatus
    add(const vector& x, const vector& y, clsparseControl control)
    {
        assert (x.size() == y.size());
        assert (x.size() == _size);

        //
        clsparseStatus status =
                elementwise_transform<T, EW_PLUS>(*this, x, y, control);
        OPENCL_V_THROW(status, "operator add");

        return status;
    }

    // z += x
    clsparseStatus add(const vector& x, clsparseControl control)
    {
       return add(*this, x, control);
    }

    //z = x - y;
    clsparseStatus sub(const vector& x, const vector& y, clsparseControl control)
    {
        assert (x.size() == y.size());
        assert (x.size() == _size);

        clsparseStatus status =
            elementwise_transform<T, EW_MINUS>(*this, x, y, control);
        OPENCL_V_THROW(status, "operator subtract");

        return status;
    }

    //z -= x;
    clsparseStatus sub(const vector& x, clsparseControl control)
    {
        return sub(*this, x, control);
    }

    //z = x*y;
    clsparseStatus mul(const vector& x, const vector& y, clsparseControl control)
    {
        assert (x.size() == y.size());
        assert (x.size() == _size);

        clsparseStatus status =
            elementwise_transform<T, EW_MULTIPLY>(*this, x, y, control);
        OPENCL_V_THROW(status, "operator multiply");

        return status;
    }

    //z *= x;
    clsparseStatus mul(const vector &x, clsparseControl control)
    {
        return mul(*this, x, control);
    }

    //z = x/y (zero division is not checked!)
    clsparseStatus div(const vector& x, const vector& y, clsparseControl control)
    {
        assert (x.size() == y.size());
        assert (x.size() == _size);

        clsparseStatus status =
            elementwise_transform<T, EW_DIV>(*this, x, y, control);
        OPENCL_V_THROW(status, "operator div");

        return status;
    }

    //z /= x (zero division is not checked
    clsparseStatus div(const vector& x, clsparseControl control)
    {
        return div(*this, x, control);
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
