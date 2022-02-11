#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(int argn, char *argv[])
{

    // Host problem definition
    int m = 4;
    int n = 5;
    int nnz = 9;
    int lda = m;

    cuDoubleComplex hA [] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(0, 0),
                       make_cuDoubleComplex(4, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
                       make_cuDoubleComplex(0, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(9, 0),
                       make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(0, 0),
                       make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(6, 0)};

    int hNnzPerCol[] = {2, 2, 2, 1, 2};

    cuDoubleComplex hCscValA[nnz];
    int hCscRowIndA[nnz];
    int hCscColPtrA[n + 1];

    cuDoubleComplex hCscValA_result[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(9, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(6, 0)};
    int hCscRowIndA_result[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    int hCscColPtrA_result[] = {0, 2, 4, 6, 7, 9};

    // Device memory management
    cuDoubleComplex *dCscValA, *dA;
    int *dCscRowIndA, *dCscColPtrA, *dNnzPerCol;

    CHECK_CUDA( cudaMalloc((void**) &dCscValA,  nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCscRowIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtrA, (n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dNnzPerCol, n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * n * sizeof(cuDoubleComplex)));

    CHECK_CUDA( cudaMemcpy(dA, hA, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dNnzPerCol, hNnzPerCol, n * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseZdense2csc(handle, m, n, descrA, dA, lda, dNnzPerCol, dCscValA, dCscRowIndA, dCscColPtrA);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCscValA, dCscValA, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCscRowIndA, dCscRowIndA, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCscColPtrA, dCscColPtrA, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCscValA[i].x - hCscValA_result[i].x) > 0.000001) || (fabs(hCscRowIndA[i] - hCscRowIndA_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (n + 1); i++) {
        if((fabs(hCscColPtrA[i] - hCscColPtrA_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("dense2csc test PASSED\n");
    else
        printf("dense2csc test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUDA(cudaFree(dCscValA) );
    CHECK_CUDA(cudaFree(dCscRowIndA) );
    CHECK_CUDA(cudaFree(dCscColPtrA) );
    CHECK_CUDA(cudaFree(dA) );
    CHECK_CUDA(cudaFree(dNnzPerCol) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}