#ifndef OCL_TYPE_TRAITS_HPP_
#define OCL_TYPE_TRAITS_HPP_

#define DECLARE_TYPE(TYPE) template<> struct OclTypeTraits<TYPE> \
    { static const char* type;};

template<typename T>
struct OclTypeTraits
{
};

DECLARE_TYPE(char)
DECLARE_TYPE(unsigned char)
DECLARE_TYPE(short)
DECLARE_TYPE(unsigned short)
DECLARE_TYPE(int)
DECLARE_TYPE(unsigned int)
DECLARE_TYPE(long unsigned int)
DECLARE_TYPE(long long)
DECLARE_TYPE(unsigned long long)
DECLARE_TYPE(float)
DECLARE_TYPE(double)

#endif // OCL_TYPE_TRAITS_HPP_
