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
    int m = 3;
    int n = 3;
    int nnz = 9;

    float hCsrVal[] = {3, 2, 1, 4, 6, 5, 8, 9, 7};
    int hCsrRowPtr[] = {0, 3, 6, 9};
    int hCsrColInd[] = {2, 1, 0, 0, 2, 1, 1, 2, 0};

    int hCsrColInd_result[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};

    // Device memory management
    float *dCsrVal;
    int *dCsrRowPtr, *dCsrColInd, *dp;

    //CHECK_CUDA( cudaMalloc((void**) &dCooValA,  nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrVal, nnz * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColInd, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dp, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrVal, hCsrVal, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtr, hCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColInd, hCsrColInd, nnz * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    CHECK_CUSPARSE(cusparseXcsrsort_bufferSizeExt(handle, m, n, nnz, dCsrRowPtr, dCsrColInd, &pBufferSizeInBytes));

    if (pBufferSizeInBytes == 0) {
        pBufferSizeInBytes = 1280;
    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSizeInBytes * sizeof(char)));

    CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, nnz, dp));

    CHECK_CUSPARSE(cusparseXcsrsort(handle, m, n, nnz, descrA, dCsrRowPtr, dCsrColInd, dp, pBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hCsrColInd, dCsrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrColInd[i] - hCsrColInd_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csrsort test PASSED\n");
    else
        printf("csrsort test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCsrColInd) );
    CHECK_CUDA(cudaFree(dCsrRowPtr) );
    CHECK_CUDA(cudaFree(dCsrVal) );
    CHECK_CUDA(cudaFree(dp) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}