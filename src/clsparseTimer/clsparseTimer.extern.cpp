/* ************************************************************************
 * Copyright 2015 Advanced Micro Devices, Inc.
 * ************************************************************************/

// StatTimer.cpp : Defines the exported functions for the DLL application.
//

#include "clsparseTimer-extern.hpp"
#include "clsparseTimer.host.hpp"
#include "clsparseTimer.device.hpp"

//	Even though the individual getInstance functions of the timer classes return references,
//	we convert those to pointers before returning from here so that the clients can initialize
//	their local variables to NULL, which references do not allow.
clsparseTimer* clsparseGetTimer(const clsparseTimerType type)
{
    if (type == CLSPARSE_CPU)
        return	&clsparseHostTimer::getInstance();

    return	&clsparseDeviceTimer::getInstance();
}
