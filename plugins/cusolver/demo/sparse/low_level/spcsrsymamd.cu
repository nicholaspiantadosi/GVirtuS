#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int n = 3;
    int nnzA = 9;
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    int p_result[] = {0, 1, 2};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int p[n];

    int correct = 1;
    cs = cusolverSpXcsrsymamdHost(handle, n, nnzA, descrA, hCsrRowPtrA, hCsrColIndA, p);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    for (int i = 0; i < n; i++) {
        printf("%d\n", p[i]);
        if (fabsf(p[i] - p_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrsymamd test PASSED\n");
    } else {
        printf("spcsrsymamd test FAILED\n");
    }

    return EXIT_SUCCESS;
}