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

gvirtus_add_backend(cusparse ${CUDA_VERSION}
        backend/CusparseHandler.cpp
		backend/CusparseHandler_management.cpp
		backend/CusparseHandler_helper.cpp
		backend/CusparseHandler_level1.cpp
		backend/CusparseHandler_level2.cpp
		backend/CusparseHandler_level3.cpp
		backend/CusparseHandler_format_conversion_reference.cpp
		backend/CusparseHandler_generic.cpp
		backend/CusparseHandler_extra.cpp
		backend/CusparseHandler_preconditioners_reference.cpp
		backend/CusparseHandler_reorderings.cpp
		backend/Utilities.h
)

target_link_libraries(${PROJECT_NAME} ${CUSPARSE_LIBRARY})

gvirtus_add_frontend(cusparse ${CUDA_VERSION}
        frontend/Cusparse.cpp
        frontend/CusparseFrontend.cpp
		frontend/Cusparse_management.cpp
		frontend/Cusparse_helper.cpp
		frontend/Cusparse_level1.cpp
		frontend/Cusparse_level2.cpp
		frontend/Cusparse_level3.cpp
		frontend/Cusparse_format_conversion_reference.cpp
		frontend/Cusparse_generic.cpp
		frontend/Cusparse_extra.cpp
		frontend/Cusparse_preconditioners_reference.cpp
		frontend/Cusparse_reorderings.cpp
		frontend/Utilities.h
)

#add_subdirectory(demo/level3)