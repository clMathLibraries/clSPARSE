#ifndef _PARAMS_H_
#define _PARAMS_H_

#include <string>

template<typename T>
class Params
{
public:
    T alpha;
    T beta;
    int number_of_tries;
    int number_of_warmups;
    std::string root_dir;
};

#endif //_PARAMS_H_
