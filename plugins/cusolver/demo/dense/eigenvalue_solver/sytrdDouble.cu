#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    double hd_result[] = {0.294117, 5.705881, 12};
    double he_result[] = {0.823529, -5.830952};
    double htau_result[] = {0, 1.857493, 0};

    double *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double *dd, *de, *dtau;
    CUDA_CHECK( cudaMalloc((void**) &dd, n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &de, (n - 1) * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dtau, n * sizeof(double)));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnDsytrd_bufferSize(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, &Lwork));

    double *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDsytrd(handle, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dd, de, dtau, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double valuesD[n];
    double valuesE[n-1];
    double valuesTau[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dd, n * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, de, (n - 1) * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTau, dtau, n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n ; i++) {
        printf("%f \t %f\n", valuesD[i], valuesTau[i]);
        if (fabsf(valuesD[i] - hd_result[i]) > 0.001
        || fabsf(valuesTau[i] - htau_result[i]) > 0.001) {
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