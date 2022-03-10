#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
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
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dB, m * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dX, m * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n  * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, m * nrhs * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, m * nrhs * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnSSgels_bufferSize(handle, m, n, nrhs, NULL, lda, NULL, ldb, NULL, ldx, NULL, &lwork_bytes));
    //printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnSSgels(handle, m, n, nrhs, dA, lda, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i], hX_result[i]);
        if (fabsf(values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            //break;
        }
    }

    if (correct == 1) {
        printf("DnSSgels test PASSED\n");
    } else {
        printf("DnSSgels test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}