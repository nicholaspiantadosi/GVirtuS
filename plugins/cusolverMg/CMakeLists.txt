cmake_minimum_required(VERSION 3.17)
project(gvirtus-plugin-cusolverMg)

find_package(CUDA REQUIRED)

find_path(CUSOLVER_INCLUDE_DIRECTORY
        cusolverMg.h
        PATHS ${CUDA_INCLUDE_DIRS})
if(NOT CUSOLVER_INCLUDE_DIRECTORY)
    message(FATAL_ERROR "cusolverMg.h not found")
endif()
include_directories(${CUSOLVER_INCLUDE_DIRECTORY} ${CUDA_INCLUDE_DIRS})

get_filename_component(CUDA_LIBRARIES_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY FALSE)
find_library(CUSOLVERMG_LIBRARY
        cusolverMg
        PATHS ${CUDA_LIBRARIES_PATH})
if(NOT CUSOLVER_LIBRARY)
    message(FATAL_ERROR "cusolver library not found")
endif()
message(STATUS "Found cuSolverMg library: ${CUSOLVERMG_LIBRARY}")

set(CUSOLVERMG_VERSION "${CUSOLVERMG_LIBRARY}")
while(IS_SYMLINK "${CUSOLVERMG_VERSION}")
    message(STATUS "${CUSOLVERMG_VERSION}")
    execute_process(COMMAND readlink "${CUSOLVERMG_VERSION}"
            OUTPUT_VARIABLE CUSOLVERMG_VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE)
endwhile()
string(REGEX REPLACE ".*\\.so\\.([0-9]+\\.[0-9]+\\.[0-9]+)" "\\1" CUSOLVERMG_VERSION "${CUSOLVERMG_VERSION}")

message(STATUS "Found cuSolverMg: ${CUSOLVERMG_VERSION} (${CUSOLVER_INCLUDE_DIRECTORY}/cusolverMg.h, ${CUSOLVERMG_LIBRARY})")

gvirtus_add_backend(cusolverMg ${CUDA_VERSION}
        backend/CusolverMgHandler.cpp
        backend/CusolverMgHandler_multi_gpu_helper.cpp
        backend/CusolverMgHandler_multi_gpu_dense_linear_solver.cpp
        backend/CusolverMgHandler_multi_gpu_dense_linear_eigenvalue.cpp)
target_link_libraries(${PROJECT_NAME} ${CUSOLVERMG_LIBRARY})

gvirtus_add_frontend(cusolverMg ${CUDA_VERSION}
        frontend/CusolverMg.cpp
        frontend/CusolverMgFrontend.cpp
        frontend/CusolverMg_multi_gpu_helper.cpp
        frontend/CusolverMg_multi_gpu_dense_linear_solver.cpp
        frontend/CusolverMg_multi_gpu_dense_linear_eigenvalue.cpp)

add_subdirectory(demo)
