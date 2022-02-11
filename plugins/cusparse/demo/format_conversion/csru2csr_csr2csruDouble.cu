#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(void)
{
    // Host problem definition
    int m = 3;
    int n = 3;
    int nnz = 9;

    double hCsrVal[] = {3, 2, 1, 4, 6, 5, 8, 9, 7};
    int hCsrRowPtr[] = {0, 3, 6, 9};
    int hCsrColInd[] = {2, 1, 0, 0, 2, 1, 1, 2, 0};

    double hCsrVal_sorted[nnz];
    int hCsrRowPtr_sorted[m + 1];
    int hCsrColInd_sorted[nnz];

    double hCsrVal_sorted_result[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtr_sorted_result[] = {0, 3, 6, 9};
    int hCsrColInd_sorted_result[] = {0, 1, 2, 0, 1, 2, 0, 1 , 2};

    double hCsrVal_unsorted[nnz];
    int hCsrRowPtr_unsorted[m + 1];
    int hCsrColInd_unsorted[nnz];

    // Device memory management
    double *dCsrVal;
    int *dCsrRowPtr, *dCsrColInd;

    CHECK_CUDA( cudaMalloc((void**) &dCsrVal, nnz * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColInd, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrVal, hCsrVal, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtr, hCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColInd, hCsrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    csru2csrInfo_t info;
    CHECK_CUSPARSE(cusparseCreateCsru2csrInfo(&info));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    size_t pBufferSize = 0;
    void *pBuffer = NULL;

    CHECK_CUSPARSE(cusparseDcsru2csr_bufferSizeExt(handle, m, n, nnz, dCsrVal, dCsrRowPtr, dCsrColInd, info, &pBufferSize));

    if (pBufferSize == 0) {
        pBufferSize = 1408;
    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize * sizeof(char)));

    CHECK_CUSPARSE(cusparseDcsru2csr(handle, m, n, nnz, descrA, dCsrVal, dCsrRowPtr, dCsrColInd, info, pBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hCsrVal_sorted, dCsrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtr_sorted, dCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColInd_sorted, dCsrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrVal_sorted[i] - hCsrVal_sorted_result[i]) > 0.000001)
            || (fabs(hCsrColInd_sorted[i] - hCsrColInd_sorted_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (m + 1); i++) {
        if((fabs(hCsrRowPtr_sorted[i] - hCsrRowPtr_sorted_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csru2csr test PASSED\n");
    else
        printf("csru2csr test FAILED: wrong result\n");

    if (correct) {
        CHECK_CUSPARSE(cusparseDcsr2csru(handle, m, n, nnz, descrA, dCsrVal, dCsrRowPtr, dCsrColInd, info, pBuffer));

        // device result check
        CHECK_CUDA( cudaMemcpy(hCsrVal_unsorted, dCsrVal, nnz * sizeof(double), cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(hCsrRowPtr_unsorted, dCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
        CHECK_CUDA( cudaMemcpy(hCsrColInd_unsorted, dCsrColInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

        correct = 1;
        for (int i = 0; i < nnz; i++) {
            if((fabs(hCsrVal_unsorted[i] - hCsrVal[i]) > 0.000001)
               || (fabs(hCsrColInd_unsorted[i] - hCsrColInd[i]) > 0.000001)) {
                correct = 0;
                break;
            }
        }
        for (int i = 0; i < (m + 1); i++) {
            if((fabs(hCsrRowPtr_unsorted[i] - hCsrRowPtr_sorted[i]) > 0.000001)) {
                correct = 0;
                break;
            }
        }

        if (correct)
            printf("csr2csru test PASSED\n");
        else
            printf("csr2csru test FAILED: wrong result\n");
    }

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCsrColInd) );
    CHECK_CUDA(cudaFree(dCsrRowPtr) );
    CHECK_CUDA(cudaFree(dCsrVal) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));
    CHECK_CUSPARSE(cusparseDestroyCsru2csrInfo(info));

    return EXIT_SUCCESS;
}
