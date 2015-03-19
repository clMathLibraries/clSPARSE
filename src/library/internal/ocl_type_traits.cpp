#include "ocl_type_traits.hpp"


#define DECLARE_TYPE_STR(TYPE, TYPE_STR) \
    const char* OclTypeTraits<TYPE>::type = TYPE_STR;

DECLARE_TYPE_STR( cl_char, "char" )
DECLARE_TYPE_STR( cl_uchar, "uchar" )
DECLARE_TYPE_STR( cl_short, "short" )
DECLARE_TYPE_STR( cl_ushort, "ushort" )
DECLARE_TYPE_STR( cl_int, "int" )
DECLARE_TYPE_STR( cl_uint, "uint" )
DECLARE_TYPE_STR( cl_long, "long" )
DECLARE_TYPE_STR( cl_ulong, "ulong" )
DECLARE_TYPE_STR( cl_float, "float" )
DECLARE_TYPE_STR( cl_double, "double" )
