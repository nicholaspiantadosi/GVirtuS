#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverRf.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../cusolver_utils.h"

int main(void) {

    cusolverRfHandle_t handle = NULL;

    cusolverStatus_t cs = cusolverRfCreate(&handle);
    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int n = 3;
    int nnzA = 9;
    double hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzL = 9;
    double hCsrValL[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrL[] = {0, 3, 6, 9};
    const int hCsrColIndL[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzU = 9;
    double hCsrValU[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrU[] = {0, 3, 6, 9};
    const int hCsrColIndU[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int hP[] = {1, 1, 1};
    int hQ[] = {1, 1, 1};
    double *dCsrValA, *dCsrValL, *dCsrValU;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrL, *dCsrColIndL, *dCsrRowPtrU, *dCsrColIndU, *dP, *dQ;
    CUDA_CHECK( cudaMalloc((void**) &dCsrValA, nnzA * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndA, nnzA * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrValL, nnzL * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrL, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndL, nnzL * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrValU, nnzU * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrU, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndU, nnzU * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dP, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dQ, n * sizeof(int)));
    CUDA_CHECK( cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrValL, hCsrValL, nnzL * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrL, hCsrRowPtrL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndL, hCsrColIndL, nnzL * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrValU, hCsrValU, nnzU * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrU, hCsrRowPtrU, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndU, hCsrColIndU, nnzU * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dP, hP, n * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dQ, hQ, n * sizeof(int), cudaMemcpyHostToDevice) );
    cs = cusolverRfSetupDevice(n, nnzA, dCsrRowPtrA, dCsrColIndA, dCsrValA, nnzL, dCsrRowPtrL, dCsrColIndL, dCsrValL, nnzU, dCsrRowPtrU, dCsrColIndU, dCsrValU, dP, dQ, handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfAnalyze(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfRefactor(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int nnzM;
    int *dMp, *dMi;
    double *dMx;
    CUDA_CHECK( cudaMalloc((void**) &dMp, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dMi, nnzM * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dMx, nnzM * sizeof(double)));
    cs = cusolverRfAccessBundledFactorsDevice(handle, &nnzM, &dMp, &dMi, &dMx);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("nnzM: %d\n", nnzM);
    correct = nnzM == 15;
    int hMp[n + 1];
    int hMi[nnzM];
    double hMx[nnzM];
    int hMp_result[] = {0, 5, 10, 15};
    int hMi_result[] = {0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2};
    double hMx_result[] = {0.000000, 0.000000, 10.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    CUDA_CHECK( cudaMemcpy(hMp, dMp, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(hMi, dMi, nnzM * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(hMx, dMx, nnzM * sizeof(double), cudaMemcpyDeviceToHost) );
    //printArray(hMp, (n+1), "Mp");
    //printArray(hMi, nnzM, "Mi");
    //printArrayD(hMx, nnzM, "Mx");
    for (int i = 0; i < (n + 1); i++) {
        if (fabsf(hMp[i] - hMp_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnzM; i++) {
        if (fabsf(hMi[i] - hMi_result[i]) > 0.001 || fabsf(hMx[i] - hMx_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    cs = cusolverRfDestroy(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    CUDA_CHECK(cudaFree(dCsrValA));
    CUDA_CHECK(cudaFree(dCsrRowPtrA));
    CUDA_CHECK(cudaFree(dCsrColIndA));
    CUDA_CHECK(cudaFree(dCsrValL));
    CUDA_CHECK(cudaFree(dCsrRowPtrL));
    CUDA_CHECK(cudaFree(dCsrColIndL));
    CUDA_CHECK(cudaFree(dCsrValU));
    CUDA_CHECK(cudaFree(dCsrRowPtrU));
    CUDA_CHECK(cudaFree(dCsrColIndU));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dQ));

    if (correct == 1) {
        printf("rf_create_destroy test PASSED\n");
    } else {
        printf("rf_create_destroy test FAILED\n");
    }

    return EXIT_SUCCESS;
}