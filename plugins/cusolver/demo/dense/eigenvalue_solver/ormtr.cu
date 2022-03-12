#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = 3;
    int ldc = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hC[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    float htau_result[] = {0, 0};

    float *dA, *dC;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dC, n * ldc * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dC, hC, n * ldc * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    float *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dtau, (m - 1) * sizeof(float)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnSormtr_bufferSize(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, m, n, dA, lda, dtau, dC, ldc, &Lwork));

    float *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSormtr(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, m, n, dA, lda, dtau, dC, ldc, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float valuesTau[m - 1];
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, (m - 1) * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < (m - 1) ; i++) {
        printf("%f\n", valuesTau[i]);
        if (fabsf(valuesTau[i] - htau_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("ormtr test PASSED\n");
    } else {
        printf("ormtr test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}