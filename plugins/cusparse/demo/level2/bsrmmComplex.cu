#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{

    // Host problem definition
    const int blockSize = 2;
    const int mb = 2;
    const int kb = 2;
    const int nnzb = 4;

    cuComplex hBsrValA[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0),
                            make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0), make_cuComplex(0, 0),
                            make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(8, 0),
                            make_cuComplex(6, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(4, 0)};
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    cuComplex hB[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(6, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(8, 0)};

    cuComplex hC[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(0, 0),
                      make_cuComplex(0, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(0, 0),
                      make_cuComplex(4, 0), make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
                      make_cuComplex(6, 0), make_cuComplex(7, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex hC_result[] = {make_cuComplex(10, 0), make_cuComplex(8, 0), make_cuComplex(27, 0), make_cuComplex(28, 0),
                             make_cuComplex(12, 0), make_cuComplex(2, 0), make_cuComplex(31, 0), make_cuComplex(16, 0),
                             make_cuComplex(32, 0), make_cuComplex(5, 0), make_cuComplex(72, 0), make_cuComplex(24, 0),
                             make_cuComplex(44, 0), make_cuComplex(7, 0), make_cuComplex(98, 0), make_cuComplex(32, 0)};

    cuComplex alpha = make_cuComplex(1, 0);
    cuComplex beta = make_cuComplex(1, 0);

    // A is mb*kb, B is k*n and C is m*n
    const int m = mb*blockSize;
    const int k = kb*blockSize;
    const int ldb = k; // leading dimension of B
    const int ldc = m; // leading dimension of C

    // Device memory management
    cuComplex *dBsrValA;
    int *dBsrRowPtrA, *dBsrColIndA;
    cuComplex *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (blockSize * blockSize) * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dB,  ldb * m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dC,  ldc * m * sizeof(cuComplex)));

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (blockSize * blockSize) * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, ldb * m * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, ldc * m * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);

    // perform C:=alpha*A*B + beta*C
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseCbsrmm(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, m, kb, nnzb, &alpha, descrA, dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, dB, ldb, &beta, dC, ldc);

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, ldc * m * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < ldc * m; i++) {
        if (hC[i].x != hC_result[i].x) { // direct cuComplexing point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("bsrmm test PASSED\n");
    else
        printf("bsrmm test FAILED: wrong result\n");

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBsrValA) );
    CHECK_CUDA( cudaFree(dBsrRowPtrA) );
    CHECK_CUDA( cudaFree(dBsrColIndA) );
    CHECK_CUDA( cudaFree(dB) );
    CHECK_CUDA( cudaFree(dC) );
    return EXIT_SUCCESS;
}