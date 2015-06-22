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
