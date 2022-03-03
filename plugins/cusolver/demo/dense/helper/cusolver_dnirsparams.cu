#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    cusolverDnHandle_t handle = NULL;
    cusolverDnIRSParams_t params = NULL;

    cusolverStatus_t cs = cusolverDnCreate(&handle);

    cs = cusolverDnIRSParamsCreate(&params);

    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsDestroy(params);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_dnirsparams test PASSED\n");
    } else {
        printf("cusolver_dnirsparams test FAILED\n");
    }

    cusolverDnDestroy(handle);

    return EXIT_SUCCESS;
}