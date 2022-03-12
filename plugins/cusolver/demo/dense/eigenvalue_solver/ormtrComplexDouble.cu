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
    cuDoubleComplex hA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};
    cuDoubleComplex hC[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};

    cuDoubleComplex htau_result[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex *dA, *dC;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dC, n * ldc * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dC, hC, n * ldc * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cuDoubleComplex *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dtau, (m - 1) * sizeof(cuDoubleComplex)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnZunmtr_bufferSize(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, m, n, dA, lda, dtau, dC, ldc, &Lwork));

    cuDoubleComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZunmtr(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, m, n, dA, lda, dtau, dC, ldc, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuDoubleComplex valuesTau[m - 1];
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, (m - 1) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < (m - 1) ; i++) {
        printf("%f\n", valuesTau[i]);
        if (fabsf(valuesTau[i].x - htau_result[i].x) > 0.001) {
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