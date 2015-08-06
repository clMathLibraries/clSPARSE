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

#include <stdio.h>

#include <clSPARSE.h>

/**
 * Sample clSPARSE initialization (C)
 * This is the simplest program using clSPARSE API calls.
 */
int main( int argc, char* argv[ ] )
{
    printf("Executing sample clSPARSE initalization C\n");

	clsparseStatus status = clsparseSetup();

    if (status != clsparseSuccess)
    {
        printf ("Problem with executing clsparseSetup()");
        return -1;
    }


    // Put the rest of clSPARSE calls between clsparseSetup / clsparseTeardown functions;


    status = clsparseTeardown();
    if (status != clsparseSuccess)
    {
        printf ("Problem with executing clsparseTeardown()");
        return -1;
    }

    printf ("Program completed\n");

    return 0;
}
