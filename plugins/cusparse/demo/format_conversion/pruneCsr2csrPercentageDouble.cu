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
    int nnzA = 9;

    double hCsrValA[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCsrRowPtrA[] = {0, 2, 4, 7, 9};
    int hCsrColIndA[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};

    float percentage = 40;

    double hCsrValC_result[] = {5, 7, 8, 9, 6};
    int hCsrRowPtrC_result[] = {0, 0, 0, 3, 5};
    int hCsrColIndC_result[] = {0, 3, 4, 2, 4};

    // Device memory management
    double *dCsrValA, *dCsrValC;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrC, *dCsrColIndC;

    CHECK_CUDA(cudaMalloc((void**) &dCsrValA,  nnzA * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrRowPtrA,  (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrColIndA,  nnzA * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (m + 1)));

    CHECK_CUDA(cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrA));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));

    cusparseMatDescr_t descrC = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL ));

    pruneInfo_t info;
    CHECK_CUSPARSE(cusparseCreatePruneInfo(&info));

    size_t pBufferSize;
    void *pBuffer = 0;

    CHECK_CUSPARSE(cusparseDpruneCsr2csrByPercentage_bufferSizeExt(handle, m, n, nnzA, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, percentage, descrC, NULL, dCsrRowPtrC, NULL, info, &pBufferSize));

    pBufferSize = 512;

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize));

    int nnzc;
    int *nnzTotalDevHostPtr = &nnzc;

    CHECK_CUSPARSE(cusparseDpruneCsr2csrNnzByPercentage(handle, m, n, nnzA, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, percentage, descrC, dCsrRowPtrC, nnzTotalDevHostPtr, info, pBuffer));

    nnzc = *nnzTotalDevHostPtr;

    CHECK_CUDA(cudaMalloc((void**)&dCsrValC, sizeof(double) * nnzc));
    CHECK_CUDA(cudaMalloc((void**)&dCsrColIndC, sizeof(int) * nnzc));

    CHECK_CUSPARSE(cusparseDpruneCsr2csrByPercentage(handle, m, n, nnzA, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, percentage, descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC, info, pBuffer));

    // device result check

    double hCsrValC[nnzc];
    int hCsrRowPtrC[m + 1];
    int hCsrColIndC[nnzc];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzc * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzc * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (nnzc != 5) {
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
        printf("pruneCsr2csrPercentage test PASSED\n");
    else
        printf("pruneCsr2csrPercentage test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(dCsrValC) );
    CHECK_CUDA(cudaFree(dCsrRowPtrC) );
    CHECK_CUDA(cudaFree(dCsrColIndC) );
    CHECK_CUDA(cudaFree(dCsrValA) );
    CHECK_CUDA(cudaFree(dCsrRowPtrA) );
    CHECK_CUDA(cudaFree(dCsrColIndA) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}