#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};

    float hD_result[] = {-3.741657, -1.573688, -0.339662};
    float hE_result[] = {14.952305, 2.415928, 0.000000};
    float hTAUQ_result[] = {1.267261, 1.654486, 0.000000};
    float hTAUP_result[] = {1.482605, 0.000000, 0.000000};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnSgebrd_bufferSize(handle, m, n, &Lwork));

    float *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    float *dD, *dE, *dTAUQ, *dTAUP;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dD, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dE, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUQ, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUP, n * sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgebrd(handle, m, n, dA, lda, dD, dE, dTAUQ, dTAUP, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float valuesD[n];
    float valuesE[n];
    float valuesTAUQ[n];
    float valuesTAUP[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dD, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, dE, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUQ, dTAUQ, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUP, dTAUP, n * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n ; i++) {
        //printf("%f \t %f \t %f \t %f\n", valuesD[i], valuesE[i], valuesTAUQ[i], valuesTAUP[i]);
        if (fabsf(valuesD[i] - hD_result[i]) > 0.001
        || fabsf(valuesE[i] - hE_result[i]) > 0.001
        || fabsf(valuesTAUQ[i] - hTAUQ_result[i]) > 0.001
        || fabsf(valuesTAUP[i] - hTAUP_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("gebrd test PASSED\n");
    } else {
        printf("gebrd test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}