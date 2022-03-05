#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = n;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hA_result[] = {3, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, m, n, dA, lda, &Lwork));

    float *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devIpiv;
    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, m * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSgetrf(handle, m, n, dA, lda, Workspace, devIpiv, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[m*n];
    CUDA_CHECK( cudaMemcpy(values, dA, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i], hA_result[i]);
        if (fabsf(values[i] - hA_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("cusolver_dngetrf test PASSED\n");
    } else {
        printf("cusolver_dngetrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}