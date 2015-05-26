#pragma once
#ifndef _CLSPARSE_CLARRAY_HPP_
#define _CLSPARSE_CLARRAY_HPP_

#include "internal/clsparse_control.hpp"
#include "include/clSPARSE-private.hpp"
#include "clsparse.error.hpp"

#include "clarray_base.hpp"
#include "reference_base.hpp"

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

    array(clsparseControl control) : _size(0), queue(control->queue) {}

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
         BASE::buff = mem;
    }


    //returns deep copy of the object
    array copy(clsparseControl control)
    {
        cl::CommandQueue queue(control->queue);

        const auto& eventWaitList = control->event_wait_list;
        cl::Event controlEvent;
        cl_int status;

        if (!_size)
        {
            //need to provide valid event
            status = queue.enqueueMarkerWithWaitList(&eventWaitList, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueMarkerWithWaitList");

            return array();
        }
        else
        {
            const cl::Buffer& src = BASE::buff;


            array new_array(control, _size);
            cl::Buffer& dst = new_array();

            status = queue.enqueueCopyBuffer(src, dst, 0, 0,
                                             sizeof(value_type) * _size,
                                             &eventWaitList, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueCopyBuffer");

            if (control->async)
            {
                control->event = controlEvent;
            }
            else
            {
                status = controlEvent.wait();
                OPENCL_V_THROW(status, "controlEvent.wait");
            }

            return new_array;
        }
    }

    cl_int fill(clsparseControl control, const T &value)
    {
        //cl::CommandQueue queue(control->queue);

        const auto& eventWaitList = control->event_wait_list;
        cl::Event controlEvent;
        cl_int status;
        if (!_size)
        {
            //need to provide valid event
            status = queue.enqueueMarkerWithWaitList(&eventWaitList, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueMarkerWithWaitList");
        }
        else
        {
            status = queue.enqueueFillBuffer(BASE::buff, value, 0,
                                             _size * sizeof(value_type),
                                             &eventWaitList, &controlEvent);
            OPENCL_V_THROW(status, "queue.enqueueFillBuffer");
        }

        if (control->async)
        {
            control->event = controlEvent;
        }
        else
        {
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

        return reference( *this, n, queue);
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
