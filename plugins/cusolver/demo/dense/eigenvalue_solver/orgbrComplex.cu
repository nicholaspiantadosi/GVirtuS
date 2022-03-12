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
    cuComplex hA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};;

    cuComplex htau_result[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0.000000, 0)};

    cuComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cuComplex *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dtau, n * sizeof(cuComplex)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnCungbr_bufferSize(handle, CUBLAS_SIDE_LEFT, m, n, k, dA, lda, dtau, &Lwork));

    cuComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCungbr(handle, CUBLAS_SIDE_LEFT, m, n, k, dA, lda, dtau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuComplex valuesTau[n];
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n ; i++) {
        printf("%f\n", valuesTau[i].x);
        if (fabsf(valuesTau[i].x - htau_result[i].x) > 0.001) {
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