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
    int ldc = n;
    cuDoubleComplex hA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};
    cuDoubleComplex hC[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};

    cuDoubleComplex hTau_result[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex *dA, *dC, *dTau;
    CUDA_CHECK( cudaMalloc((void**) &dA, lda * k * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dC, ldc * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dTau, m * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, lda * k  * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dC, hC, ldc * n  * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnZunmqr_bufferSize(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, k, dA, lda, dTau, dC, ldc, &Lwork));

    cuDoubleComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZunmqr(handle, CUBLAS_SIDE_LEFT, CUBLAS_OP_N, m, n, k, dA, lda, dTau, dC, ldc, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuDoubleComplex values[m*n];
    CUDA_CHECK( cudaMemcpy(values, dTau, m * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i].x, hTau_result[i].x);
        if (fabsf(values[i].x - hTau_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("Dnormqr test PASSED\n");
    } else {
        printf("Dnormqr test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}