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
    cuComplex left_bottom_corner = make_cuComplex(0, 0);
    cuComplex right_upper_corner = make_cuComplex(10, 10);

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    int num_eigs;

    int correct = 1;
    cs = cusolverSpScsreigsHost(handle, m, nnz, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, left_bottom_corner, right_upper_corner, &num_eigs);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%d\n", num_eigs);
    correct = num_eigs == 1;

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsreigs test PASSED\n");
    } else {
        printf("spcsreigs test FAILED\n");
    }

    return EXIT_SUCCESS;
}