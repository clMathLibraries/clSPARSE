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

#include <iostream>

#include <clSPARSE.h>

/**
 * Sample clSPARSE initialization (C++)
 * This is the simplest program using clSPARSE API calls.
 */

int main(int argc, char* argv[])
{
    std::cout << "Executing sample clSPARSE initalization C++" << std::endl;

	clsparseStatus status = clsparseSetup();
    if (status != clsparseSuccess)
    {
        std::cerr << "Problem with executing clsparseSetup()" << std::endl;
        return -1;
    }


    // Put the rest of clSPARSE calls between clsparseSetup / clsparseTeardown functions;


	status = clsparseTeardown();
    if (status != clsparseSuccess)
    {
        std::cerr << "Problem with executing clsparseTeardown()" << std::endl;
        return -2;
    }

    std::cout << "Program completed" << std::endl;
	return 0;
}
