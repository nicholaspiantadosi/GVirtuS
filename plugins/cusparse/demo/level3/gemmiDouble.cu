#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{
    // Host problem definition

    int m = 4;
    int n = 5;
    int k = 5;
    int nnz = 9;
    double alpha = 1;
    double hA[] = {
            1, 0, 2, 3, 0,
            0, 4, 0, 0, 0,
            5, 0, 6, 0, 7,
            0, 8, 0, 9, 0 };
    int lda = m;
    double hCscValB[] = {1, 5, 4, 2, 3, 9, 7, 8, 6};
    int hCscColPtrB[] = {0, 2, 4, 6, 7, 9};
    int hCscRowIndB[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    double beta = 1;
    double hC[] = {
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0 };
    int ldc = m;
    double hC_result[] = {
            1, 0, 27, 3, 4,
            0, 16, 12, 54, 0,
            75, 0, 0, 0, 35,
            0, 36, 0, 82, 0 };

    // Device memory management
    double *dA, *dC;
    double *dCscValB;
    int *dCscColPtrB, *dCscRowIndB;

    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * k * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCscValB, nnz * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtrB, (k + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscRowIndB, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dC,  ldc * n * sizeof(double)));

    CHECK_CUDA( cudaMemcpy(dA, hA, lda * k * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscValB, hCscValB, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscColPtrB, hCscColPtrB, (k + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscRowIndB, hCscRowIndB, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, ldc * n * sizeof(double), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK_CUSPARSE(cusparseDgemmi(handle, m, n, k, nnz, &alpha, dA, lda, dCscValB, dCscColPtrB, dCscRowIndB, &beta, dC, ldc));

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, ldc * n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < ldc * n; i++) {
        if((fabs(hC[i] - hC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("gemmi test PASSED\n");
    else
        printf("gemmi test FAILED: wrong result\n");

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA( cudaFree(dA) );
    CHECK_CUDA( cudaFree(dCscValB) );
    CHECK_CUDA( cudaFree(dCscColPtrB) );
    CHECK_CUDA( cudaFree(dCscRowIndB) );
    CHECK_CUDA( cudaFree(dC) );
    return EXIT_SUCCESS;

}