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
    int ldx = n;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hX[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    float hX_result[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    float *dA, *dB, *dX;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dX, n * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n  * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, n * nrhs * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnSBgesv_bufferSize(handle, n, nrhs, NULL, lda, NULL, NULL, ldb, NULL, ldx, NULL, &lwork_bytes));
    //lwork_bytes = 157568;
    printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    int *devIpiv;
    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSBgesv(handle, n, nrhs, dA, lda, devIpiv, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i], hX_result[i]);
        if (fabsf(values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("DnSBgesv test PASSED\n");
    } else {
        printf("DnSBgesv test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}