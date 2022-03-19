#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int m = 3;
    int n = 3;
    int nnz = 9;
    double hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    double b[] = {1, 1, 1};

    double x_result[] = {0.097473, 0.155235, -0.014440};
    int p_result[] = {0, 1, 2};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    double x[n];
    int p[n];
    int rankA;
    double min_norm;

    int correct = 1;
    cs = cusolverSpDcsrlsqvqrHost(handle, m, n, nnz, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, b, 1, &rankA, x, p, &min_norm);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%d\n", rankA);
    correct = rankA == 3;
    correct = min_norm == 0;

    for (int i = 0; i < n; i++) {
        printf("%f \t %d\n", x[i], p[i]);
        if (fabsf(x[i] - x_result[i]) > 0.01 || fabsf(p[i] - p_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrlsqvqr test PASSED\n");
    } else {
        printf("spcsrlsqvqr test FAILED\n");
    }

    return EXIT_SUCCESS;
}