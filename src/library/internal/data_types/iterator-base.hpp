#pragma once
#ifndef _CLSPARSE_ITERATOR_BASE_HPP_
#define _CLSPARSE_ITERATOR_BASE_HPP_


#include "include/clSPARSE-private.hpp"
#include "clSPARSE-error.hpp"
#include "reference_base.hpp"
#include <cassert>
#include <iterator>

namespace clsparse
{

/*  This is just a test code. It works, but don't know if there are some performance issues;
 *  I did not distincted the functions for forward, reverse, constant etc.
 *  Treat this as a base line for something more mature
 */

template <typename Container>
class iterator_base
{
public:

    typedef iterator_base self_type;

    typedef typename Container::value_type value_type;
    typedef reference_base<Container> reference;

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

    reference operator*()
    {
        return reference( container, index, queue);
    }

//    const reference operator*()
//    {
//        return reference( container, index, queue);
//    }

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

    iterator_base < Container >& operator++ ()
    {
        index += 1;
        return *this;
    }


    iterator_base < Container >& operator++ (int i)
    {
        index += 1;
        return *this;
    }

    iterator_base < Container >& operator-- ()
    {
        index -= 1;
        return *this;
    }


    iterator_base < Container >& operator-- (int i)
    {
        index -= 1;
        return *this;
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
    cl::CommandQueue& queue;

}; //iterator_base

} //namespace clsparse

#endif //_CLSPARSE_ITERATOR_BASE_HPP_
