#include "clsparse_internal.h"

// flag indicating initialization of the library
int clsparseInitialized = 0;

//list of program sources
hdl_list* program_sources = NULL;
//list of compiled kernels
hdl_list* kernel_cache = NULL;
