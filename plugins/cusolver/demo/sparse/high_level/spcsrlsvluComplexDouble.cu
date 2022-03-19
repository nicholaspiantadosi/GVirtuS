#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int n = 3;
    int nnzA = 9;
    cuDoubleComplex hCsrValA[] = {make_cuDoubleComplex(10, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(9, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(-6, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(2, 0)};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    cuDoubleComplex b[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};

    cuDoubleComplex x_result[] = {make_cuDoubleComplex(0.097473, 0), make_cuDoubleComplex(0.155235, 0), make_cuDoubleComplex(-0.014440, 0)};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cuDoubleComplex x[n];
    int singularity;

    int correct = 1;
    cs = cusolverSpZcsrlsvluHost(handle, n, nnzA, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, b, 1, 0, x, &singularity);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%d\n", singularity);
    correct = singularity == -1;

    for (int i = 0; i < n; i++) {
        printf("%f\n", x[i].x);
        if (fabsf(x[i].x - x_result[i].x) > 0.01) {
            correct = 0;
            break;
        }
    }

    cusolverSpDestroy(handle);

    if (correct == 1) {
        printf("spcsrlsvlu test PASSED\n");
    } else {
        printf("spcsrlsvlu test FAILED\n");
    }

    return EXIT_SUCCESS;
}