#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int nrhs = 3;
    int lda = n;
    int ldb = n;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hB_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    double *dA, *dB;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int *devIpiv;
    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs, dA, lda, devIpiv, dB, ldb, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dB, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * n; i++) {
        printf("%f == %f\n", values[i], hB_result[i]);
        if (fabsf(values[i] - hB_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("cusolverDnSgetrs test PASSED\n");
    } else {
        printf("cusolverDnSgetrs test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}