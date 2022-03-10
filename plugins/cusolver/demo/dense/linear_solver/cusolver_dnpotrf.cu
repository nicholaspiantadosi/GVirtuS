#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hA_result[] = {1, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, dA, lda, &Lwork));

    float *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, dA, lda, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*lda];
    CUDA_CHECK( cudaMemcpy(values, dA, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * lda; i++) {
        if (fabsf(values[i] - hA_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("cusolver_dnpotrf test PASSED\n");
    } else {
        printf("cusolver_dnpotrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}