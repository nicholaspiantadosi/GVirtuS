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
    float hTAU_result[] = {1.267261, 1.801257, 0, 0, 0, 0, 0, 0, 0};

    float *dA, *dTAU;
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dTAU, m * n * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    CUSOLVER_CHECK(cusolverDnXgeqrf_bufferSize(handle, params, m, n, CUDA_R_32F, dA, lda, CUDA_R_32F, dTAU, CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void *bufferOnDevice;
    cudaMalloc((void**)&bufferOnDevice, workspaceInBytesOnDevice);
    size_t bufferOnHost[workspaceInBytesOnHost];

    int *info;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnXgeqrf(handle, params, m, n, CUDA_R_32F, dA, lda, CUDA_R_32F, dTAU, CUDA_R_32F, bufferOnDevice, workspaceInBytesOnDevice, &bufferOnHost, workspaceInBytesOnHost, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[m * n];
    CUDA_CHECK( cudaMemcpy(values, dTAU, m * n * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i], hTAU_result[i]);
        if (fabsf(values[i] - hTAU_result[i]) > 0.05) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dnxgeqrf test PASSED\n");
    } else {
        printf("dnxgeqrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}