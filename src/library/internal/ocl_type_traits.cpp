#include "ocl_type_traits.hpp"


#define DECLARE_TYPE_STR(TYPE, TYPE_STR) \
    const char* OclTypeTraits<TYPE>::type = TYPE_STR;

DECLARE_TYPE_STR(char, "char")
DECLARE_TYPE_STR(unsigned char, "uchar")
DECLARE_TYPE_STR(short, "short")
DECLARE_TYPE_STR(unsigned short, "ushort")
DECLARE_TYPE_STR(int, "int")
DECLARE_TYPE_STR(unsigned int, "uint")
DECLARE_TYPE_STR(long long, "long")
DECLARE_TYPE_STR(unsigned long long, "ulong")
DECLARE_TYPE_STR(float, "float")
DECLARE_TYPE_STR(double, "double")
