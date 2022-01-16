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
    cuComplex alpha = make_cuComplex(1, 0);
    cuComplex hA[] = {
            make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(6, 0), make_cuComplex(0, 0), make_cuComplex(7, 0),
            make_cuComplex(0, 0), make_cuComplex(8, 0), make_cuComplex(0, 0), make_cuComplex(9, 0), make_cuComplex(0, 0) };
    int lda = m;
    cuComplex hCscValB[] = {make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(4, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(9, 0),
                            make_cuComplex(7, 0), make_cuComplex(8, 0), make_cuComplex(6, 0) };
    int hCscColPtrB[] = {0, 2, 4, 6, 7, 9};
    int hCscRowIndB[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    cuComplex beta = make_cuComplex(1, 0);
    cuComplex hC[] = {
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0) };
    int ldc = m;
    cuComplex hC_result[] = {
            make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(27, 0), make_cuComplex(3, 0), make_cuComplex(4, 0),
            make_cuComplex(0, 0), make_cuComplex(16, 0), make_cuComplex(12, 0), make_cuComplex(54, 0), make_cuComplex(0, 0),
            make_cuComplex(75, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(35, 0),
            make_cuComplex(0, 0), make_cuComplex(36, 0), make_cuComplex(0, 0), make_cuComplex(82, 0), make_cuComplex(0, 0) };

    // Device memory management
    cuComplex *dA, *dC;
    cuComplex *dCscValB;
    int *dCscColPtrB, *dCscRowIndB;

    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * k * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCscValB, nnz * sizeof(cuComplex)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtrB, (k + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscRowIndB, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dC,  ldc * n * sizeof(cuComplex)));

    CHECK_CUDA( cudaMemcpy(dA, hA, lda * k * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscValB, hCscValB, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscColPtrB, hCscColPtrB, (k + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscRowIndB, hCscRowIndB, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, ldc * n * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    CHECK_CUSPARSE(cusparseCgemmi(handle, m, n, k, nnz, &alpha, dA, lda, dCscValB, dCscColPtrB, dCscRowIndB, &beta, dC, ldc));

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, ldc * n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < ldc * n; i++) {
        if((fabs(hC[i].x - hC_result[i].x) > 0.000001)) {
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