#ifndef _CLSPARSE_INTERNAL_H_
#define _CLSPARSE_INTERNAL_H_

#include "hdl_list.h"

#ifdef __cplusplus
extern "C" {
#endif

extern int clsparseInitialized;

extern hdl_list* program_sources;
extern hdl_list* kernel_cache;


//void createSourcesMap(void);

#ifdef __cplusplus
}
#endif

#endif //_CLSPARSE_INTERNAL_H_
