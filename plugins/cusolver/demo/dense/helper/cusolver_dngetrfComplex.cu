#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = n;
    cuComplex hA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};
    cuComplex hA_result[] = {make_cuComplex(3, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnCgetrf_bufferSize(handle, m, n, dA, lda, &Lwork));

    cuComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devIpiv;
    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, m * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCgetrf(handle, m, n, dA, lda, Workspace, devIpiv, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuComplex values[m*n];
    CUDA_CHECK( cudaMemcpy(values, dA, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i].x, hA_result[i].x);
        if (fabsf(values[i].x - hA_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("cusolver_dngetrf test PASSED\n");
    } else {
        printf("cusolver_dngetrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}