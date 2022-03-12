#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    cuDoubleComplex hA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};

    double hd_result[] = {0.294117, 5.705881, 12};
    double he_result[] = {0.823529, -5.830952};
    cuDoubleComplex htau_result[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1.857493, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double *dd, *de;
    cuDoubleComplex *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dd, n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &de, (n - 1) * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dtau, n * sizeof(cuDoubleComplex)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnZhetrd_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, &Lwork));

    cuDoubleComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZhetrd(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double valuesD[n];
    double valuesE[n-1];
    cuDoubleComplex valuesTau[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dd, n * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, de, (n - 1) * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

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