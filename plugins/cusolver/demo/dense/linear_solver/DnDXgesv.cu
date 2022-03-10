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
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hX[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    double hX_result[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    double *dA, *dB, *dX;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dX, n * nrhs * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n  * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, n * nrhs * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnDXgesv_bufferSize(handle, n, nrhs, NULL, lda, NULL, NULL, ldb, NULL, ldx, NULL, &lwork_bytes));
    //lwork_bytes = 88832;
    printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    int *devIpiv;
    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDXgesv(handle, n, nrhs, dA, lda, devIpiv, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i], hX_result[i]);
        if (fabsf(values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("DnDXgesv test PASSED\n");
    } else {
        printf("DnDXgesv test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}