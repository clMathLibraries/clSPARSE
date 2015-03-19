#ifndef OCL_TYPE_TRAITS_HPP_
#define OCL_TYPE_TRAITS_HPP_

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#define DECLARE_TYPE(TYPE) template<> struct OclTypeTraits<TYPE> \
    { static const char* type;};

template<typename T>
struct OclTypeTraits
{
};

DECLARE_TYPE( cl_char )
DECLARE_TYPE( cl_uchar )
DECLARE_TYPE( cl_short )
DECLARE_TYPE( cl_ushort )
DECLARE_TYPE( cl_int )
DECLARE_TYPE( cl_uint )
DECLARE_TYPE( cl_long )
DECLARE_TYPE( cl_ulong )
DECLARE_TYPE( cl_float )
DECLARE_TYPE( cl_double )

#endif // OCL_TYPE_TRAITS_HPP_
