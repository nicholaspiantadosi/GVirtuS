#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "../level2/utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{

    // Host problem definition
    const int blockSize = 2;
    const int mb = 2;
    const int kb = 2;
    const int nnzb = 4;

    double hBsrValA[] = {1, 0, 0, 4, 2, 0, 3, 0, 5, 0, 0, 8, 6, 0, 7, 4};
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    double hB[] = {1, 2, 0, 3, 0, 0, 0, 4, 0, 0, 5, 6, 0, 0, 7, 8};

    double hC[] = {0, 0, 1, 0, 0, 2, 3, 0, 4, 5, 0, 0, 6, 7, 0, 0};
    double hC_result[] = {10, 8, 27, 28, 12, 2, 31, 16, 32, 5, 72, 24, 44, 7, 98, 32};

    double alpha = 1;
    double beta = 1;

    // A is mb*kb, B is k*n and C is m*n
    const int m = mb*blockSize;
    const int k = kb*blockSize;
    const int ldb = k; // leading dimension of B
    const int ldc = m; // leading dimension of C

    // Device memory management
    double *dBsrValA;
    int *dBsrRowPtrA, *dBsrColIndA;
    double *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (blockSize * blockSize) * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dB,  ldb * m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dC,  ldc * m * sizeof(double)));

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (blockSize * blockSize) * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, ldb * m * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, ldc * m * sizeof(double), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);

    // perform C:=alpha*A*B + beta*C
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );
    cusparseDbsrmm(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, m, kb, nnzb, &alpha, descrA, dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, dB, ldb, &beta, dC, ldc);

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, ldc * m * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < ldc * m; i++) {
        if (hC[i] != hC_result[i]) { // direct doubleing point comparison is not
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