#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int nrhs = 3;
    int lda = 3;
    int ldb = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    float hB_result[] = {-0.142857, 0.142857, 0.285714, 2.171429, -0.085714, 0, 2.542857, -0.2, 0.285714};

    float *dA, *dB;
    CUDA_CHECK( cudaMalloc((void**) &dA, lda * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dB, ldb * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, lda * n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, ldb * nrhs * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int64_t hipiv[] = {1, 0, 0};
    int64_t *dipiv;
    CUDA_CHECK( cudaMalloc((void**) &dipiv, n * sizeof(int64_t)));
    CUDA_CHECK( cudaMemcpy(dipiv, hipiv, n * sizeof(int64_t), cudaMemcpyHostToDevice) );

    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
    CUSOLVER_CHECK(cusolverDnXsytrs_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, CUDA_R_32F, dA, lda, dipiv, CUDA_R_32F, dB, ldb, &workspaceInBytesOnDevice, &workspaceInBytesOnHost));

    void *bufferOnDevice;
    cudaMalloc((void**)&bufferOnDevice, workspaceInBytesOnDevice);
    size_t bufferOnHost[workspaceInBytesOnHost];

    int *info;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnXsytrs(handle, CUBLAS_FILL_MODE_LOWER, n, nrhs, CUDA_R_32F, dA, lda, dipiv, CUDA_R_32F, dB, ldb, bufferOnDevice, workspaceInBytesOnDevice, &bufferOnHost, workspaceInBytesOnHost, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[ldb * nrhs];
    CUDA_CHECK( cudaMemcpy(values, dB, ldb * nrhs * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < ldb * nrhs; i++) {
        printf("%f == %f\n", values[i], hB_result[i]);
        if (fabsf(values[i] - hB_result[i]) > 0.05) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dnxsytrs test PASSED\n");
    } else {
        printf("dnxsytrs test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}