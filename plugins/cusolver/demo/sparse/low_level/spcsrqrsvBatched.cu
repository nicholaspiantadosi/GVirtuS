#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int m = 3;
    int n = 3;
    int nnzA = 9;
    int batchSize = 5;
    float hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2, 9, 1, 9, 3, 4, -6, 1, 6, 2, 8, 1, 9, 3, 4, -6, 1, 6, 2, 7, 1, 9, 3, 4, -6, 1, 6, 2, 5, 1, 9, 3, 4, -6, 1, 6, 2};
    int hCsrRowPtrA[] = {0, 3, 6, 9};
    int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    float x_result[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    csrqrInfo_t info = NULL;
    cusolverSpCreateCsrqrInfo(&info);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int correct = 1;
    cs = cusolverSpXcsrqrAnalysisBatched(handle, m, n, nnzA, descrA, hCsrRowPtrA, hCsrColIndA, info);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    size_t internalDataInBytes;
    size_t workspaceInBytes;
    cs = cusolverSpScsrqrBufferInfoBatched(handle, m, n, nnzA, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, batchSize, info, &internalDataInBytes, &workspaceInBytes);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    void *pBuffer = 0;
    cudaMalloc((void**)&pBuffer, workspaceInBytes);

    float x[m*batchSize];
    float b[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    cs = cusolverSpScsrqrsvBatched(handle, m, n, nnzA, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, b, x, batchSize, info, pBuffer);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    for (int i = 0; i < m*batchSize; i++) {
        printf("%f\n", x[i]);
        if (fabsf(x[i] - x_result[i]) > 0.01) {
            correct = 0;
            //break;
        }
    }

    cusolverSpDestroyCsrqrInfo(info);
    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrqrsvBatched test PASSED\n");
    } else {
        printf("spcsrqrsvBatched test FAILED\n");
    }

    return EXIT_SUCCESS;
}