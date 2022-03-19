#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int n = 3;
    int nnzA = 9;
    double hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    int P_result[] = {0, 1, 2};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int P[n];
    int numnz;

    int correct = 1;
    cs = cusolverSpDcsrzfdHost(handle, n, nnzA, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, P, &numnz);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%d\n", numnz);
    correct = numnz == 3;

    for (int i = 0; i < n; i++) {
        printf("%d\n", P[i]);
        if (fabsf(P[i] - P_result[i]) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrzfd test PASSED\n");
    } else {
        printf("spcsrzfd test FAILED\n");
    }

    return EXIT_SUCCESS;
}