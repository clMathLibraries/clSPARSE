/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
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

 /*! \file
 * \brief Abstracts platform differences to loading dynamic libraries at runtime
 * \details The entire implementation is written in the header file, and provides functions
 * to load, unload and create a function address pointer to an exported library function
 */

#pragma once
#ifndef _SHAREDLIBRARY_H_
#define _SHAREDLIBRARY_H_
#include <string>

//	_WIN32 is defined for both 32 & 64 bit environments
#if defined( _WIN32 )
#define WIN32_LEAN_AND_MEAN			// Exclude rarely-used stuff from Windows headers
// Windows Header Files:
#include <windows.h>
#else
#include <dlfcn.h>
#endif

/*!
* \brief Create a platform specific handle to a loaded dynamic library
* \details Calls platform specific code to load a dynamic library and create a handle for it
* \param[in] libPrefix  Prefix to be appended on linux style OS's
* \param[in] libraryName  Name of the library of interest
* \param[in] quiet  Print error information to the CONSOLE if true
*
* \returns Platform specific library handle
*
*/
inline void* LoadSharedLibrary( const std::string& libPrefix, std::string libraryName, bool quiet )
{
#if defined( _WIN32 )
    libraryName += ".dll";

    //	HMODULE is actually the load address; function returns NULL if it cannot find the shared library
    HMODULE fileHandle = ::LoadLibraryExA(libraryName.c_str(), NULL, NULL);
#elif defined(__linux__)
    std::string linuxName = libPrefix;
    linuxName += libraryName += ".so";
    void* fileHandle = ::dlopen( linuxName.c_str( ), RTLD_NOW );
    if( !quiet && !fileHandle )
    {
        std::cerr << ::dlerror( ) << std::endl;
    }
#elif defined(__APPLE__)
    std::string appleName = libPrefix;
    appleName += libraryName += ".dylib";
    void* fileHandle = ::dlopen( appleName.c_str( ), RTLD_NOW );
    if( !quiet && !fileHandle )
    {
        std::cerr << ::dlerror( ) << std::endl;
    }
#else
#error "unsupported platform"
#endif

    return fileHandle;
}

/*!
* \brief Release the handle to the dynamic library
* \details Calls platform specific code to release the handle to a dynamic library
* \param[in,out] libHandle  Platform handle to the dynamic library, NULL'd on output
*
* \returns If the function succeeds, return value is nonzero.  If the function fails, return value is zero.
*
*/
inline int FreeSharedLibrary(void*& libHandle)
{
    int result = 0;

#if defined( _WIN32 )
    if (libHandle != 0)
        result = ::FreeLibrary(reinterpret_cast<HMODULE>(libHandle));
#else
    if( libHandle != 0 )
        result = ( ::dlclose( libHandle ) == 0 );
#endif

    libHandle = NULL;

    return result;
}

/*!
 * \brief Query for function pointer in library
 * \details This takes a shared module handle returned from LoadSharedLibrary, and a text string of a symbol
 * to load from the module, and returns a pointer to that symbol.  If the symbol is not found, NULL
 * is returned.  If the module handle is NULL, NULL is returned.
 * \param[in] libHandle  Platform handle to the dynamic library
 * \param[in] funcName  String representing the function name of interest
 *
 * \returns Function pointer (or NULL) to the function of interest in the dynamic library
 *
 */
inline void* LoadFunctionAddr(void* libHandle, std::string funcName)
{
    if (libHandle == NULL)
        return NULL;

#if defined( _WIN32 )
    HMODULE fileHandle = reinterpret_cast<HMODULE>(libHandle);

    void* pFunc = reinterpret_cast<void*>(::GetProcAddress(fileHandle, funcName.c_str()));
#else
    void* pFunc = ::dlsym( libHandle, funcName.c_str( ) );
#endif

    return pFunc;
}

#endif // _SHAREDLIBRARY_H_
