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

    cuComplex hTAU_result[] = {make_cuComplex(1.267261, 0), make_cuComplex(1.801257, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex *dA, *dTAU;
    CUDA_CHECK( cudaMalloc((void**) &dA, m * n * sizeof(cuComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dTAU, m * n * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, m * n  * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnCgeqrf_bufferSize(handle, m, n, dA, lda, &Lwork));

    cuComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCgeqrf(handle, m, n, dA, lda, dTAU, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuComplex values[m*n];
    CUDA_CHECK( cudaMemcpy(values, dTAU, m * n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < m * n; i++) {
        printf("%f == %f\n", values[i].x, hTAU_result[i].x);
        if (fabsf(values[i].x - hTAU_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("Dngeqrf test PASSED\n");
    } else {
        printf("Dngeqrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}