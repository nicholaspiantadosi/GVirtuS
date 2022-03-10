#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    int ldb = 3;
    double hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    double hX_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    double *dA, *dB;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * ldb * sizeof(double)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * ldb * sizeof(double), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int *devInfo;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnDpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, ldb, dA, lda, dB, ldb, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double values[n*ldb];
    CUDA_CHECK( cudaMemcpy(values, dB, sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * ldb; i++) {
        printf("%f == %f\n", values[i], hX_result[i]);
        if (fabsf(values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("cusolver_dnpotrs test PASSED\n");
    } else {
        printf("cusolver_dnpotrs test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}