#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    cuComplex hA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};

    float hd_result[] = {0.294117, 5.705881, 12};
    float he_result[] = {0.823529, -5.830952};
    cuComplex htau_result[] = {make_cuComplex(0, 0), make_cuComplex(1.857493, 0), make_cuComplex(0, 0)};

    cuComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    float *dd, *de;
    cuComplex *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dd, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &de, (n - 1) * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dtau, n * sizeof(cuComplex)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnChetrd_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, &Lwork));

    cuComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnChetrd(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float valuesD[n];
    float valuesE[n-1];
    cuComplex valuesTau[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dd, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, de, (n - 1) * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n ; i++) {
        printf("%f \t %f\n", valuesD[i], valuesTau[i]);
        if (fabsf(valuesD[i] - hd_result[i]) > 0.001
        || fabsf(valuesTau[i].x - htau_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (n - 1) ; i++) {
        printf("%f\n", valuesE[i]);
        if (fabsf(valuesE[i] - he_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("sytrd test PASSED\n");
    } else {
        printf("sytrd test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}