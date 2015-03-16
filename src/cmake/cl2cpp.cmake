MESSAGE(STATUS "running cl2cpp")

file(GLOB cl_list "${CL_DIR}/*.cl" )

file(WRITE ${OUTPUT} "// This file is auto-generated. Do not edit!

#include \"internal/source_provider.hpp\"

namespace internal
{
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

  file(APPEND ${OUTPUT} "const char* ${cl_filename}=\"${lines};\n")
endforeach()


file(APPEND ${OUTPUT} "}\n")


file(APPEND ${OUTPUT} "

std::map<std::string, const char*> SourceProvider::map(SourceProvider::MapInit());

std::map<std::string, const char*> SourceProvider::MapInit()
{
	std::map<std::string, const char*> internal_map;
")

foreach(cl ${cl_list})
  get_filename_component(cl_filename "${cl}" NAME_WE)

  file(APPEND ${OUTPUT} "
	internal_map[\"${cl_filename}\"] = internal::${cl_filename};\n")
endforeach()




file(APPEND ${OUTPUT} "
	return internal_map;
}
")

