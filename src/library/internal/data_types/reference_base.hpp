#pragma once
#ifndef _CLSPARSE_REFERENCE_BASE_HPP_
#define _CLSPARSE_REFERENCE_BASE_HPP_

#include "include/clSPARSE-private.hpp"

namespace clsparse
{
// exact bolt implementation for map/unmap helper
template< typename Container >
class reference_base
{
public:

    typedef typename Container::value_type value_type;
    typedef value_type* naked_pointer;
    typedef const value_type* const_naked_pointer;

    reference_base(Container &rhs, size_t index , cl::CommandQueue queue):
        container( rhs ), index( index ), queue(queue)
    {}

    //  Automatic type conversion operator to turn the reference object into a value_type
    operator value_type() const
    {
        cl_int status = CL_SUCCESS;
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer( container.buffer(), true, CL_MAP_READ,
                                            index * sizeof( value_type ),
                                            sizeof( value_type ),
                                            NULL, NULL, &status)
                    );

        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        value_type valTmp = *result;

        ::cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.buffer(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return valTmp;
    }

    reference_base< Container >& operator=(const value_type& rhs )
    {
        cl_int status = CL_SUCCESS;
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer(container.buffer(), true, CL_MAP_WRITE_INVALIDATE_REGION,
                                           index * sizeof( value_type ), sizeof( value_type ),
                                           NULL, NULL, &status ) );
        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        *result = rhs;

        ::cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.buffer(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        OPENCL_V_THROW( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return *this;
    }

    reference_base< Container >& operator=(reference_base< Container >& rhs )
    {

        cl_int status = CL_SUCCESS;
        value_type value = static_cast<value_type>(rhs);
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer(container.buffer(), true, CL_MAP_WRITE_INVALIDATE_REGION,
                                           index * sizeof( value_type ), sizeof( value_type ),
                                           NULL, NULL, &status) );
        OPENCL_V_THROW( status, "Array failed map device memory to host memory for operator[]" );

        *result = value;

        cl::Event unmapEvent;
        OPENCL_V_THROW( queue.enqueueUnmapMemObject( container.buffer(), result, NULL, &unmapEvent ),
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
    size_t index;
    cl::CommandQueue queue;

}; //reference_base

} //namespace clsparse

#endif

