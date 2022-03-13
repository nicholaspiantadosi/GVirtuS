#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    float hA_result[] = {1, -0.4, -0.083333, 2, 0.2, -0.083333, 3, 5, 0.083333};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, lda * n * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, lda * n * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    CUSOLVER_CHECK(cusolverDnXtrtri_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT, n, CUDA_R_32F, dA, lda, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void *bufferOnDevice;
    cudaMalloc((void**)&bufferOnDevice, workspaceInBytesOnDevice);
    size_t bufferOnHost[workspaceInBytesOnHost];

    int *info;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnXtrtri(handle, CUBLAS_FILL_MODE_LOWER, CUBLAS_DIAG_NON_UNIT, n, CUDA_R_32F, dA, lda, bufferOnDevice, workspaceInBytesOnDevice, &bufferOnHost, workspaceInBytesOnHost, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[lda * n];
    CUDA_CHECK( cudaMemcpy(values, dA, lda * n * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < lda * n; i++) {
        printf("%f == %f\n", values[i], hA_result[i]);
        if (fabsf(values[i] - hA_result[i]) > 0.05) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dnxtrtri test PASSED\n");
    } else {
        printf("dnxtrtri test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}