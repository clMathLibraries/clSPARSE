// This file is auto-generated. Do not edit!
// File contains kernel sources generated from *.cl files.
// Sources have the same name as the cl files.
// Each source code have it's own key created as <filename>_key
// In the code they will build a map {source_key : kernel_source }

#ifndef _PROGRAM_SOURCES_H_
#define _PROGRAM_SOURCES_H_

#ifdef __cplusplus
extern "C" {
#endif

const char* scale_key="scale";
const char* scale="//TEST KERNEL\n"
"\n"
"#ifndef INDEX_TYPE\n"
"#error INDEX_TYPE undefined!\n"
"#endif\n"
"\n"
"#ifndef VALUE_TYPE\n"
"#error VALUE_TYPE undefined!\n"
"#endif\n"
"\n"
"#ifndef SIZE_TYPE\n"
"#error SIZE_TYPE undefined!\n"
"#endif\n"
"\n"
"// v = v*alpha\n"
"__kernel\n"
"__attribute__((reqd_work_group_size(WG_SIZE, 1, 1)))\n"
"void scale ( __global VALUE_TYPE* v,\n"
"             __global VALUE_TYPE* alpha,\n"
"             const SIZE_TYPE size)\n"
"{\n"
"    const int i = get_global_id(0);\n"
"\n"
"    if (i >= size) return;\n"
"\n"
"    v[i] = v[i]* alpha[0];\n"
"}\n"
"\n"
;


#ifdef __cplusplus
}
#endif
#endif //_PROGRAM_SOURCES_H
