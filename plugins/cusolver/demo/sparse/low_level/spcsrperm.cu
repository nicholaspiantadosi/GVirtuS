#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int m = 3;
    int n = 3;
    int nnzA = 9;
    int hCsrRowPtrA[] = {0, 3, 6, 9};
    int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    int p[] = {0, 1, 2};
    int q[] = {0, 1, 2};

    int hCsrRowPtrA_result[] = {0, 3, 6, 9};
    int hCsrColIndA_result[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    size_t bufferSizeInBytes;

    int correct = 1;
    cs = cusolverSpXcsrperm_bufferSizeHost(handle, m, n, nnzA, descrA, hCsrRowPtrA, hCsrColIndA, p, q, &bufferSizeInBytes);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    size_t pBuffer[bufferSizeInBytes];
    int map[nnzA];

    cs = cusolverSpXcsrpermHost(handle, m, n, nnzA, descrA, hCsrRowPtrA, hCsrColIndA, p, q, map, pBuffer);

    for (int i = 0; i < m+1; i++) {
        printf("%d\n", hCsrRowPtrA[i]);
        if (fabsf(hCsrRowPtrA[i] - hCsrRowPtrA_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    for (int i = 0; i < nnzA; i++) {
        printf("%d\n", hCsrColIndA[i]);
        if (fabsf(hCsrColIndA[i] - hCsrColIndA_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrperm test PASSED\n");
    } else {
        printf("spcsrperm test FAILED\n");
    }

    return EXIT_SUCCESS;
}