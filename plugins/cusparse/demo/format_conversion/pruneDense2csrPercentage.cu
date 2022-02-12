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
    int lda = m;

    float hA [] = {1, 0, 5, 0,
                    4, 2, 0, 0,
                    0, 3, 0, 9,
                    0, 0, 7, 0,
                    0, 0, 8, 6};

    float percentage = 90;

    float hCsrValC_result[] = {8, 9};
    int hCsrRowPtrC_result[] = {0, 0, 0, 1, 2};
    int hCsrColIndC_result[] = {4, 2};

    // Device memory management
    float *dA;
    float *dCsrValC;
    int *dCsrRowPtrC, *dCsrColIndC;

    CHECK_CUDA(cudaMalloc((void**) &dA,  m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (m + 1)));

    CHECK_CUDA(cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrC = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL ));

    pruneInfo_t info;
    CHECK_CUSPARSE(cusparseCreatePruneInfo(&info));

    size_t pBufferSize;
    void *pBuffer = 0;

    CHECK_CUSPARSE(cusparseSpruneDense2csrByPercentage_bufferSizeExt(handle, m, n, dA, lda, percentage, descrC, NULL, dCsrRowPtrC, NULL, info, &pBufferSize));

    if(pBufferSize == 0) {
        pBufferSize = 512;
    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize));

    int nnzc;
    int *nnzTotalDevHostPtr = &nnzc;

    CHECK_CUSPARSE(cusparseSpruneDense2csrNnzByPercentage(handle, m, n, dA, lda, percentage, descrC, dCsrRowPtrC, nnzTotalDevHostPtr, info, pBuffer));

    nnzc = *nnzTotalDevHostPtr;

    CHECK_CUDA(cudaMalloc((void**)&dCsrValC, sizeof(float) * nnzc));
    CHECK_CUDA(cudaMalloc((void**)&dCsrColIndC, sizeof(int) * nnzc));

    CHECK_CUSPARSE(cusparseSpruneDense2csrByPercentage(handle, m, n, dA, lda, percentage, descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC, info, pBuffer));

    // device result check

    float hCsrValC[nnzc];
    int hCsrRowPtrC[m + 1];
    int hCsrColIndC[nnzc];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzc * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzc * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (nnzc != 2) {
        correct = 0;
    }
    for (int i = 0; i < nnzc; i++) {
        if((fabs(hCsrValC[i] - hCsrValC_result[i]) > 0.000001) || (fabs(hCsrColIndC[i] - hCsrColIndC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (m + 1); i++) {
        if((fabs(hCsrRowPtrC[i] - hCsrRowPtrC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("pruneDense2csrPercentage test PASSED\n");
    else
        printf("pruneDense2csrPercentage test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(dCsrValC) );
    CHECK_CUDA(cudaFree(dCsrRowPtrC) );
    CHECK_CUDA(cudaFree(dCsrColIndC) );
    CHECK_CUDA(cudaFree(dA) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}