/* ************************************************************************
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
