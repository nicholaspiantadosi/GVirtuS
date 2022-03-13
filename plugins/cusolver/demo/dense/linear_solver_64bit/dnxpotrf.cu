#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    CUSOLVER_CHECK(cusolverDnXpotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, dA, lda, CUDA_R_32F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    float *bufferOnDevice;
    cudaMalloc((void**)&bufferOnDevice, workspaceInBytesOnDevice);
    float bufferOnHost[workspaceInBytesOnHost];

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnXpotrf(handle, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, dA, lda, CUDA_R_32F, bufferOnDevice, workspaceInBytesOnDevice, &bufferOnHost, workspaceInBytesOnHost, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);

    if (correct == 1) {
        printf("dnxpotrf test PASSED\n");
    } else {
        printf("dnxpotrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}