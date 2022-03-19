#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    int m = 3;
    int nnz = 9;
    cuComplex hCsrValA[] = {make_cuComplex(10, 0), make_cuComplex(1, 0), make_cuComplex(9, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(-6, 0), make_cuComplex(1, 0), make_cuComplex(6, 0), make_cuComplex(2, 0)};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    const cuComplex x0[] = {make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0)};

    cuComplex x_result[] = {make_cuComplex(0.512307, 0), make_cuComplex(-0.048724, 0), make_cuComplex(-0.857419, 0)};

    cusolverSpHandle_t handle = NULL;
    cusolverStatus_t cs = cusolverSpCreate(&handle);

    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);

    cuComplex x[m];
    cuComplex mu;

    int correct = 1;
    cs = cusolverSpCcsreigvsiHost(handle, m, nnz, descrA, hCsrValA, hCsrRowPtrA, hCsrColIndA, make_cuComplex(1, 0), x0, 50, 1, &mu, x);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    printf("%f\n", mu.x);
    correct = fabsf(mu.x - 3.934861) < 0.01;

    for (int i = 0; i < m; i++) {
        printf("%f\n", x[i].x);
        if (fabsf(x[i].x - x_result[i].x) > 0.01) {
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