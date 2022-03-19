#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int m = 3;
    int nnz = 9;
    float hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const float x0[] = {1, 1, 1};

    float x_result[] = {0.512307, -0.048724, -0.857419};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    float x[m];
    float mu;

    int correct = 1;
    cs = cusolverSpScsreigvsiHost(handle, m, nnz, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, 1, x0, 50, 1, &mu, x);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%f\n", mu);
    correct = fabsf(mu - 3.934861) < 0.01;

    for (int i = 0; i < m; i++) {
        printf("%f\n", x[i]);
        if (fabsf(x[i] - x_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsreigvsi test PASSED\n");
    } else {
        printf("spcsreigvsi test FAILED\n");
    }

    return EXIT_SUCCESS;
}