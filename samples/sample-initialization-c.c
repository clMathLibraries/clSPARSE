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
