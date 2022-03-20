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
    int hCsrRowPtrA[] = {0, 3, 6, 9};
    int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzL = 9;
    double hCsrValL[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    int hCsrRowPtrL[] = {0, 3, 6, 9};
    int hCsrColIndL[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzU = 9;
    double hCsrValU[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    int hCsrRowPtrU[] = {0, 3, 6, 9};
    int hCsrColIndU[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int hP[] = {1, 1, 1};
    int hQ[] = {1, 1, 1};
    cs = cusolverRfSetupHost(n, nnzA, hCsrRowPtrA, hCsrColIndA, hCsrValA, nnzL, hCsrRowPtrL, hCsrColIndL, hCsrValL, nnzU, hCsrRowPtrU, hCsrColIndU, hCsrValU, hP, hQ, handle);
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
    int *hMp;
    int *hMi;
    double *hMx;
    cs = cusolverRfExtractBundledFactorsHost(handle, &nnzM, &hMp, &hMi, &hMx);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    printf("nnzM: %d\n", nnzM);
    correct = nnzM == 15;
    int hMp_result[] = {0, 5, 10, 15};
    int hMi_result[] = {0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2};
    double hMx_result[] = {0.000000, 0.000000, 10.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    printArray(hMp, (n+1), "Mp");
    printArray(hMi, nnzM, "Mi");
    printArrayD(hMx, nnzM, "Mx");
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

    if (correct == 1) {
        printf("rf_create_destroy_host test PASSED\n");
    } else {
        printf("rf_create_destroy_host test FAILED\n");
    }

    return EXIT_SUCCESS;
}