#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int k = 3;
    int lda = 3;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    double htau_result[] = {0, 0, 0};

    double *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dtau, n * sizeof(double)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnDorgbr_bufferSize(handle, CUBLAS_SIDE_LEFT, m, n, k, dA, lda, dtau, &Lwork));

    double *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDorgbr(handle, CUBLAS_SIDE_LEFT, m, n, k, dA, lda, dtau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double valuesTau[n];
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n ; i++) {
        printf("%f\n", valuesTau[i]);
        if (fabsf(valuesTau[i] - htau_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("orgbr test PASSED\n");
    } else {
        printf("orgbr test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}