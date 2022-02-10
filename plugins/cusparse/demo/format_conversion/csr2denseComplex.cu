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
    int n = 4;
    int nnz = 9;
    int lda = m;

    cuComplex hCsrVal[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(5, 0), make_cuComplex(6, 0), make_cuComplex(7, 0), make_cuComplex(8, 0), make_cuComplex(9, 0)};
    int hCsrRowPtr[] = {0, 3, 4, 7, 9};
    int hCsrColInd[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    cuComplex hA[m * n];

    cuComplex hA_result[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(0, 0),
                             make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(0, 0), make_cuComplex(8, 0),
                             make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(6, 0), make_cuComplex(0, 0),
                             make_cuComplex(3, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(9, 0)};

    // Device memory management
    cuComplex *dCsrVal, *dA;
    int *dCsrRowPtr, *dCsrColInd;

    CHECK_CUDA( cudaMalloc((void**) &dCsrVal,  nnz * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColInd, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * n * sizeof(cuComplex)));

    CHECK_CUDA( cudaMemcpy(dCsrVal, hCsrVal, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtr, hCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColInd, hCsrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseCcsr2dense(handle, m, n, descrA, dCsrVal, dCsrRowPtr, dCsrColInd, dA, lda);

    // device result check
    CHECK_CUDA( cudaMemcpy(hA, dA, lda * n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < lda * n; i++) {
        if((fabs(hA[i].x - hA_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csr2dense test PASSED\n");
    else
        printf("csr2dense test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUDA(cudaFree(dCsrVal) );
    CHECK_CUDA(cudaFree(dCsrRowPtr) );
    CHECK_CUDA(cudaFree(dCsrColInd) );
    CHECK_CUDA(cudaFree(dA) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}