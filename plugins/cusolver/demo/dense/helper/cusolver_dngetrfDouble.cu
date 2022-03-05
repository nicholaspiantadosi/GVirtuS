#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = n;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hA_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    double *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, m, n, dA, lda, &Lwork));

    double *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devIpiv;
    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, m * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDgetrf(handle, m, n, dA, lda, Workspace, devIpiv, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double values[m*n];
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