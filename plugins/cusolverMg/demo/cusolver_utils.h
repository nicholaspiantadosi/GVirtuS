#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <library_types.h>

// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

void printArray(int array[], int size, std::string name) {
    printf("%s: ", name.c_str());
    for (int i = 0; i < size; i++) {
        if (i == 0) {
            printf("[");
        }
        printf("%d", array[i]);
        if (i < size - 1) {
            printf(", ");
        } else {
            printf("]\n");
        }
    }
}

void printArrayD(double array[], int size, std::string name) {
    printf("%s: ", name.c_str());
    for (int i = 0; i < size; i++) {
        if (i == 0) {
            printf("[");
        }
        printf("%f", array[i]);
        if (i < size - 1) {
            printf(", ");
        } else {
            printf("]\n");
        }
    }
}