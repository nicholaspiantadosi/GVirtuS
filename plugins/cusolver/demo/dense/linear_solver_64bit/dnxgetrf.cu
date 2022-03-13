#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hA_result[] = {3, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    CUSOLVER_CHECK(cusolverDnXgetrf_bufferSize(handle, params, m, n, CUDA_R_32F, dA, lda, CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void *bufferOnDevice;
    cudaMalloc((void**)&bufferOnDevice, workspaceInBytesOnDevice);
    size_t bufferOnHost[workspaceInBytesOnHost];

    int *info;
    int64_t *devIpiv;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, m * sizeof(int64_t)));
    CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, m, n, CUDA_R_32F, dA, lda, devIpiv, CUDA_R_32F, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*lda];
    CUDA_CHECK( cudaMemcpy(values, dA, sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < n * lda; i++) {
        printf("%f == %f\n", values[i], hA_result[i]);
        if (fabsf(values[i] - hA_result[i]) > 0.05) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dnxgetrf test PASSED\n");
    } else {
        printf("dnxgetrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}