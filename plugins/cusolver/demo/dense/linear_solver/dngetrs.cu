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
    float hB_result[] = {-0.4, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA, *dB;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    int *info;
    int64_t *devIpiv;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int64_t)));
    CUSOLVER_CHECK(cusolverDnGetrs(handle, params, CUBLAS_OP_N, n, nrhs, CUDA_R_32F, dA, lda, devIpiv, CUDA_R_32F, dB, ldb, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dB, sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i], hB_result[i]);
        if (fabsf(values[i] - hB_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dngetrs test PASSED\n");
    } else {
        printf("dngetrs test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}