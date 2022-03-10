#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = n;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    int hIpiv_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, lda * n * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, lda * n  * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnSsytrf_bufferSize(handle, n, dA, lda, &Lwork));

    float *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    int *devIpiv;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSsytrf(handle, CUBLAS_FILL_MODE_LOWER, n, dA, lda, devIpiv, Workspace, Lwork, devInfo));
    int hdevInfo;
    int hIpiv[n];
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(&hIpiv, devIpiv, n * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n; i++) {
        printf("%d == %d\n", hIpiv[i], hIpiv_result[i]);
        if (fabsf(hIpiv[i] - hIpiv_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("Dnsytrf test PASSED\n");
    } else {
        printf("Dnsytrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}