#ifndef _SOURCE_PROVIDER_H_
#define _SOURCE_PROVIDER_H_

#include <string>
#include <map>

/**
 * @brief The SourceProvider class provides the source of the kernel
 * assumes that there is one kernel per cl file, kernel name = program name*
 */
class SourceProvider
{

public:
    static const char* GetSource(const std::string& name)
    {
        auto source = map.find(name);
        if (source != map.end())
        {
            return source->second;
        }
        else
        {
            return nullptr;
        }

    }

private:
    static std::map<std::string, const char*> map;

    /** actually, implementation of MapInit is generated automatically
     *  by cmake script */
    static std::map<std::string, const char*> MapInit();
};


#endif //_SOURCE_PROVIDER_H_

