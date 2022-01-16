cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cusparse)
find_package(CUDA REQUIRED)

find_path(CUSPARSE_INCLUDE_DIRECTORY
        cusparse.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUSPARSE_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cusparse.h not found")
endif()
include_directories(${CUSPARSE_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUSPARSE_LIBRARY
        cusparse
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUSPARSE_LIBRARY)
    message(FATAL_ERROR "cusparse library not found")
endif()
message(STATUS "Found cuSparse library: ${CUSPARSE_LIBRARY}")

set(CUSPARSE_VERSION "${CUSPARSE_LIBRARY}")
while(IS_SYMLINK "${CUSPARSE_VERSION}")
    message(STATUS "${CUSPARSE_VERSION}")
    execute_process(COMMAND readlink "${CUSPARSE_VERSION}"
            OUTPUT_VARIABLE CUSPARSE_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUSPARSE_VERSION "${CUSPARSE_VERSION}")

message(STATUS "Found cuSparse: ${CUSPARSE_VERSION} (${CUSPARSE_INCLUDE_DIRECTORY}/cusparse.h, ${CUSPARSE_LIBRARY})")
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(utilities.h)
cuda_add_executable(gemmi.e gemmi.cu OPTIONS --cudart=shared -lcusparse)
#set_target_properties(get_error_string.e PROPERTIES CUDA_RUNTIME_LIBRARY shared EXTRA_NVCCFLAGS --cudart=shared)
install(TARGETS gemmi.e RUNTIME DESTINATION ${GVIRTUS_HOME}/demo/cusparse/level3)
message("gemmi added")