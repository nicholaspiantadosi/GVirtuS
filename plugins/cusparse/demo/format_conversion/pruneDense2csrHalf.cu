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

    __half hA [] = {1, 0, 5, 0,
                    4, 2, 0, 0,
                    0, 3, 0, 9,
                    0, 0, 7, 0,
                    0, 0, 8, 6};

    __half threshold = 2;

    //__half hCsrValC_result[] = {4, 3, 5, 7, 8, 9, 6};
    int hCsrRowPtrC_result[] = {0, 1, 2, 5, 7};
    int hCsrColIndC_result[] = {1, 2, 0, 3, 4, 2, 4};

    // Device memory management
    __half *dA;
    __half *dCsrValC;
    int *dCsrRowPtrC, *dCsrColIndC;

    CHECK_CUDA(cudaMalloc((void**) &dA,  m * n * sizeof(__half)));
    CHECK_CUDA(cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (m + 1)));

    CHECK_CUDA(cudaMemcpy(dA, hA, m * n * sizeof(__half), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrC = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL ));

    size_t pBufferSize;
    void *pBuffer = 0;

    CHECK_CUSPARSE(cusparseHpruneDense2csr_bufferSizeExt(handle, m, n, dA, lda, &threshold, descrC, NULL, dCsrRowPtrC, NULL, &pBufferSize));

    if(pBufferSize == 0) {
        pBufferSize = 512;
    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize));

    int nnzc;
    int *nnzTotalDevHostPtr = &nnzc;

    CHECK_CUSPARSE(cusparseHpruneDense2csrNnz(handle, m, n, dA, lda, &threshold, descrC, dCsrRowPtrC, nnzTotalDevHostPtr, pBuffer));

    nnzc = *nnzTotalDevHostPtr;

    CHECK_CUDA(cudaMalloc((void**)&dCsrValC, sizeof(__half) * nnzc));
    CHECK_CUDA(cudaMalloc((void**)&dCsrColIndC, sizeof(int) * nnzc));

    CHECK_CUSPARSE(cusparseHpruneDense2csr(handle, m, n, dA, lda, &threshold, descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC, pBuffer));

    // device result check

    __half hCsrValC[nnzc];
    int hCsrRowPtrC[m + 1];
    int hCsrColIndC[nnzc];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzc * sizeof(__half), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzc * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (nnzc != 7) {
        correct = 0;
    }
    for (int i = 0; i < nnzc; i++) {
        if((fabs(hCsrColIndC[i] - hCsrColIndC_result[i]) > 0.000001)) {
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
        printf("pruneDense2csr test PASSED\n");
    else
        printf("pruneDense2csr test FAILED: wrong result\n");

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