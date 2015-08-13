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

        CLSPARSE_V( status, "Mapping device buffer on host failed" );
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
            CLSPARSE_V( queue.enqueueUnmapMemObject( container.data(), host_buffer, NULL, &unmapEvent ),
                            "Array failed to unmap host buffer back to device memory" );
            CLSPARSE_V( unmapEvent.wait( ), "Failed to wait for unmap event" );
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

        CLSPARSE_V( status, "Array failed map device memory to host memory for operator[]" );

        value_type valTmp = *result;

        ::cl::Event unmapEvent;
        CLSPARSE_V( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        CLSPARSE_V( unmapEvent.wait( ), "Failed to wait for unmap event" );

        return valTmp;
    }

    reference_base< Container >& operator=(const value_type& rhs )
    {
        cl_int status = CL_SUCCESS;
        naked_pointer result = reinterpret_cast< naked_pointer >(
                    queue.enqueueMapBuffer(container.data(), true, CL_MAP_WRITE_INVALIDATE_REGION,
                                           index * sizeof( value_type ), sizeof( value_type ),
                                           NULL, NULL, &status ) );
        CLSPARSE_V( status, "Array failed map device memory to host memory for operator[]" );

        *result = rhs;

        ::cl::Event unmapEvent;
        CLSPARSE_V( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        CLSPARSE_V( unmapEvent.wait( ), "Failed to wait for unmap event" );

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
        CLSPARSE_V( status, "Array failed map device memory to host memory for operator[]" );

        *result = value;

        cl::Event unmapEvent;
        CLSPARSE_V( queue.enqueueUnmapMemObject( container.data(), result, NULL, &unmapEvent ),
                        "Array failed to unmap host memory back to device memory" );
        CLSPARSE_V( unmapEvent.wait( ), "Failed to wait for unmap event" );

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
