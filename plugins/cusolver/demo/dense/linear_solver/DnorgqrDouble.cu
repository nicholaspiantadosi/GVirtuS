#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int k = 3;
    int lda = n;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    double hTau_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    double *dA, *dTau;
    CUDA_CHECK( cudaMalloc((void**) &dA, lda * k * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dTau, m * n * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, lda * k  * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(handle, m, n, k, dA, lda, dTau, &Lwork));

    double *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDorgqr(handle, m, n, k, dA, lda, dTau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double values[m*n];
    CUDA_CHECK( cudaMemcpy(values, dTau, m * n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i], hTau_result[i]);
        if (fabsf(values[i] - hTau_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("Dnorgqr test PASSED\n");
    } else {
        printf("Dnorgqr test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}