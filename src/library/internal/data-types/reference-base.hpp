#pragma once
#ifndef _CLSPARSE_REFERENCE_BASE_HPP_
#define _CLSPARSE_REFERENCE_BASE_HPP_

#include "include/clSPARSE-private.hpp"
#include <cassert>

namespace clsparse
{


/**
  reference_base implements host reflection of device buffer.
 */

template< typename Container >
class reference_base
{
public:

    typedef typename Container::value_type value_type;
    typedef value_type* naked_pointer;
    typedef const value_type* const_naked_pointer;
    typedef size_t difference_type;

    reference_base(Container &rhs, difference_type index,
                   difference_type range, cl::CommandQueue queue):
        container( rhs ), index( index ), range ( range ), queue(queue)
    {
        cl_int status = CL_SUCCESS;


        //should we throw or map until container.size()?
        assert( (index + range) < container.size() );

        host_buffer = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer( container.data(), true, CL_MAP_READ | CL_MAP_WRITE,
                                            index * sizeof( value_type ),
                                            range * sizeof( value_type ),
                                            NULL, NULL, &status)
                    );

        OPENCL_V_THROW( status, "Mapping device buffer on host failed" );
    }


    reference_base(Container &rhs, difference_type index,cl::CommandQueue queue):
        container( rhs ), index( index ), range ( 1 ), host_buffer( nullptr ), queue(queue)
    {
        //this will still keeps the functionallity of reference to single value
    }

    // update the device memory when reference is out of scope
    ~reference_base()
    {
        if (host_buffer)
        {
            ::cl::Event unmapEvent;
            OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.data(), host_buffer, NULL, &unmapEvent ),
                            "Array failed to unmap host buffer back to device memory" );
            OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );
        }
    }

    //  Automatic type conversion operator to turn the reference object into a value_type
    operator value_type() const
    {
        cl_int status = CL_SUCCESS;
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer( container.data(), true, CL_MAP_READ,
                                            index * sizeof( value_type ),
                                            sizeof( value_type ),
                                            NULL, NULL, &status)
                    );

        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        value_type valTmp = *result;

        ::cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return valTmp;
    }

    reference_base< Container >& operator=(const value_type& rhs )
    {
        cl_int status = CL_SUCCESS;
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer(container.data(), true, CL_MAP_WRITE_INVALIDATE_REGION,
                                           index * sizeof( value_type ), sizeof( value_type ),
                                           NULL, NULL, &status ) );
        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        *result = rhs;

        ::cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return *this;
    }

    reference_base< Container >& operator=(reference_base< Container >& rhs )
    {

        cl_int status = CL_SUCCESS;
        value_type value = static_cast<value_type>(rhs);
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer(container.data(), true, CL_MAP_WRITE_INVALIDATE_REGION,
                                           index * sizeof( value_type ), sizeof( value_type ),
                                           NULL, NULL, &status) );
        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        *result = value;

        cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return *this;
    }

    Container& getContainer( ) const
    {
        return container;
    }

    size_t getIndex() const
    {
    return index;
    }

private:

    Container& container;
    difference_type index;
    difference_type range;
    naked_pointer* host_buffer;
    cl::CommandQueue queue;

}; //reference_base

} //namespace clsparse

#endif

