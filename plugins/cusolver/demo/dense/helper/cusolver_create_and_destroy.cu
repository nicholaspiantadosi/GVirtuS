#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    cusolverDnHandle_t handle = NULL;
    cudaStream_t streamIn = NULL;
    cudaStream_t streamOut = NULL;

    cusolverStatus_t cs = cusolverDnCreate(&handle);

    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnSetStream(handle, streamIn);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnGetStream(handle, &streamOut);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnDestroy(handle);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_create_and_destroy test PASSED\n");
    } else {
        printf("cusolver_create_and_destroy test FAILED\n");
    }

    cudaStreamDestroy(streamIn);
    cudaStreamDestroy(streamOut);

    return EXIT_SUCCESS;
}