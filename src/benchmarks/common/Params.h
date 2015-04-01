#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <string>

#include "opencl_utils.h"

template<typename T>
class Params
{
public:
    T alpha;
    T beta;
    int number_of_tries;
    int number_of_warmups;
    std::string root_dir;

    cl_platform_type pID;
    cl_uint dID;

};

#endif //_PARAMS_H_
