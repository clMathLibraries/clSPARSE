# ########################################################################
# Copyright 2015 Vratis, Ltd.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ########################################################################

MESSAGE(STATUS "running cl2c")

file(GLOB cl_list "${CL_DIR}/*.cl" )


file(WRITE ${OUTPUT} "// This file is auto-generated. Do not edit!
// File contains kernel sources generated from *.cl files.
// Sources have the same name as the cl files.
// Each source code have it's own key created as <filename>_key
// In the code they will build a map {source_key : kernel_source }

#ifndef _PROGRAM_SOURCES_H_
#define _PROGRAM_SOURCES_H_

#ifdef __cplusplus
extern \"C\" {
#endif

")

foreach(cl ${cl_list})
  get_filename_component(cl_filename "${cl}" NAME_WE)
  #message("${cl_filename}")

  file(READ "${cl}" lines)

  string(REPLACE "\r" "" lines "${lines}\n")
  string(REPLACE "\t" "  " lines "${lines}")

#  string(REGEX REPLACE "/\\*([^*]/|\\*[^/]|[^*/])*\\*/" ""   lines "${lines}") # multiline comments
#  string(REGEX REPLACE "/\\*([^\n])*\\*/"               ""   lines "${lines}") # single-line comments
#  string(REGEX REPLACE "[ ]*//[^\n]*\n"                 "\n" lines "${lines}") # single-line comments
#  string(REGEX REPLACE "\n[ ]*(\n[ ]*)*"                "\n" lines "${lines}") # empty lines & leading whitespace
#  string(REGEX REPLACE "^\n"                            ""   lines "${lines}") # leading new line

  string(REPLACE "\\" "\\\\" lines "${lines}")
  string(REPLACE "\"" "\\\"" lines "${lines}")
  string(REPLACE "\n" "\\n\"\n\"" lines "${lines}")

  string(REGEX REPLACE "\"$" "" lines "${lines}") # unneeded " at the eof

  file(APPEND ${OUTPUT} "const char* ${cl_filename}_key=\"${cl_filename}\";\n")
  file(APPEND ${OUTPUT} "const char* ${cl_filename}=\"${lines};\n")
  #file(APPEND ${OUTPUT} "hdl_insert(${PS_LIST}, ${cl_filename}_key, ${cl_filename});\n\n")
endforeach()

#You can add something here
file(APPEND ${OUTPUT} "

#ifdef __cplusplus
}
#endif
#endif //_PROGRAM_SOURCES_H
")
