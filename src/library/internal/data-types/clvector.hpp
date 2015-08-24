/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

#pragma once
#ifndef _CLSPARSE_CLVECTOR_HPP_
#define _CLSPARSE_CLVECTOR_HPP_

#include "internal/clsparse-control.hpp"
#include "include/clSPARSE-private.hpp"
#include "clSPARSE-error.h"

#include "clarray-base.hpp"
#include "reference-base.hpp"
#include "iterator-base.hpp"



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
    typedef typename BASE::BUFF_TYPE BUFF_TYPE;

public:

    typedef typename BASE::value_type value_type;
    typedef typename BASE::size_type  size_type;

    typedef value_type* naked_pointer;

    typedef vector<value_type> self_type;

    typedef reference_base<self_type> reference;

    typedef iterator_base<self_type> iterator;

    vector(clsparseControl control, size_t size, const value_type& value = value_type(),
          cl_mem_flags flags = CL_MEM_READ_WRITE, cl_bool init = true) :
        queue(control->queue)
    {
        BASE::data() = create_buffer(size, flags);

        if (init)
        {
            cl_int status = fill(control, value);
            CLSPARSE_V(status, "vector.fill");
        }
    }

    //create vector from preinitialized buffer.
    vector (clsparseControl control, const cl::Buffer& buffer, size_t size)
        : BASE::_buff(buffer), BASE::_size(size), queue(control->queue)
    {}

    vector (clsparseControl control, const cl_mem& mem, size_t size)
        : queue(control->queue)
    {

        //operator = on Memory class transfers ownership which I want to avoid;
         clRetainMemObject(mem);
         BASE::_buff = mem;
         BASE::_size = size;
    }


    vector(const vector& other, bool copy = true) :
        BASE::_size(other.size()), queue(other.queue)
    {
        cl_int status;
        cl::Event controlEvent;

        const BUFF_TYPE& src = other.data();
        BUFF_TYPE& dst = BASE::data();

        cl_mem_flags flags = src.getInfo<CL_MEM_FLAGS>(&status);

        CLSPARSE_V(status, "Vector cpy constr, getInfo<CL_MEM_FLAGS>");

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

    cl_int fill(clsparseControl control, const T &value)
    {
        return internal::fill(*this, value);
    }

    size_type size() const
    {
        return BASE::_size;
    }

    void resize(size_type size)
    {
        if(this->size() != size)
        {
            cl_mem_flags flags;
            cl_int status = BASE::data().getInfo(CL_MEM_FLAGS, &flags);
            CLSPARSE_V(status, "buffer get info flags");

            BASE::data() = create_buffer(size, flags);
        }
    }

    vector shallow_copy()
    {
        vector tmpCopy;
        tmpCopy._size = size();
        tmpCopy.data() = BASE::data();
        return tmpCopy;
    }

    void clear()
    {
        BASE::clear();
    }


    reference operator[]( size_type n )
    {
        assert(n < size());

        return reference( *this, n, queue);
    }

    // returns constant value instead of reference
    // since it is not a mutable object
    const value_type operator[]( size_t n ) const
    {
        assert(n < size());

        cl_int clStatus;

        naked_pointer buffer = reinterpret_cast< naked_pointer >
                ( queue.enqueueMapBuffer(BASE::_buff, CL_TRUE, CL_MAP_READ,
                                         n * sizeof( value_type ), sizeof( value_type ),
                                         NULL, NULL, &clStatus));
        CLSPARSE_V(clStatus, "clVector failed to map value to host memory");

        const value_type retValue = *buffer;

        cl::Event unmapEvent;
        clStatus = queue.enqueueUnmapMemObject(BASE::_buff, buffer, NULL, &unmapEvent);

        CLSPARSE_V(clStatus, "clVector failed to unmap pointer from device mem");
        clStatus = unmapEvent.wait();
        CLSPARSE_V(clStatus, "clVector unmap event failed");

        return retValue;
    }


    iterator begin()
    {
        return iterator(*this, 0, queue);
    }

    iterator end()
    {
        return iterator(*this, size(), queue);
    }


    //assignment operator performs deep copy
    vector& operator= (const vector& other)
    {
        if (this != &other)
        {
            assert(other.size() > 0);

            if (size() != other.size())
                resize(other.size());

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
        return clsparse::internal::add(*this, x, y, control);
    }

    // z += x
    clsparseStatus add(const vector& x, clsparseControl control)
    {
       return add(*this, x, control);
    }

    //z = x - y;
    clsparseStatus sub(const vector& x, const vector& y, clsparseControl control)
    {
        return clsparse::internal::sub(*this, x, y, control);
    }

    //z -= x;
    clsparseStatus sub(const vector& x, clsparseControl control)
    {
        return sub(*this, x, control);
    }

    //z = x*y;
    clsparseStatus mul(const vector& x, const vector& y, clsparseControl control)
    {
        return clsparse::internal::mul(*this, x, y, control);
    }

    //z *= x;
    clsparseStatus mul(const vector &x, clsparseControl control)
    {
        return mul(*this, x, control);
    }

    //z = x/y (zero division is not checked!)
    clsparseStatus div(const vector& x, const vector& y, clsparseControl control)
    {
        clsparse::internal::div(*this, x, y, control);
    }

    //z /= x (zero division is not checked
    clsparseStatus div(const vector& x, clsparseControl control)
    {
        return div(*this, x, control);
    }

    cl::CommandQueue& getQueue()
    {
        return queue;
    }

private:

    cl::Buffer create_buffer(size_type size, cl_map_flags flags = CL_MEM_READ_WRITE)
        {
            if(size > 0)
            {
                BASE::_size = size;
                cl::Context ctx = queue.getInfo<CL_QUEUE_CONTEXT>();
                return ::cl::Buffer(ctx, flags, sizeof(value_type) * BASE::_size);
            }

			return cl::Buffer();
        }


    cl::CommandQueue& queue;

}; // vector

} //namespace clsparse


#endif //_CLSPARSE_CLVECTOR_HPP_
