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

    float hCsrVal[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtr[] = {0, 3, 4, 7, 9};
    int hCsrColInd[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    float hCscVal[nnz];
    int hCscColPtr[n + 1];
    int hCscRowInd[nnz];

    float hCscVal_result[] = {1, 5, 4, 8, 2, 6, 3, 7, 9};
    int hCscColPtr_result[] = {0, 2, 4, 6, 9};
    int hCscRowInd_result[] = {0, 2, 1, 3, 0, 2, 0, 2, 3};

    // Device memory management

    float *dCsrVal, *dCscVal;
    int *dCsrRowPtr, *dCsrColInd, *dCscColPtr, *dCscRowInd;

    CHECK_CUDA( cudaMalloc((void**) &dCsrVal, nnz * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColInd, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscVal, nnz * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtr, (n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscRowInd, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrVal, hCsrVal, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtr, hCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColInd, hCsrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSize;
    void *buffer = 0;

    cusparseStatus_t cs = cusparseCsr2cscEx2_bufferSize(handle, m, n, nnz, dCsrVal, dCsrRowPtr, dCsrColInd, dCscVal, dCscColPtr, dCscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, &bufferSize);

    cudaMalloc((void**)&buffer, bufferSize);

    cs = cusparseCsr2cscEx2(handle, m, n, nnz, dCsrVal, dCsrRowPtr, dCsrColInd, dCscVal, dCscColPtr, dCscRowInd, CUDA_R_32F, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1, buffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCscVal, dCscVal, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCscColPtr, dCscColPtr, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCscRowInd, dCscRowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCscVal[i] - hCscVal_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
        if((fabs(hCscRowInd[i] - hCscRowInd_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (n + 1); i++) {
        if((fabs(hCscColPtr[i] - hCscColPtr_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csr2cscex2 test PASSED\n");
    else
        printf("csr2cscex2 test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCsrVal));
    CHECK_CUDA(cudaFree(dCsrRowPtr));
    CHECK_CUDA(cudaFree(dCsrColInd));
    CHECK_CUDA(cudaFree(dCscVal));
    CHECK_CUDA(cudaFree(dCscColPtr));
    CHECK_CUDA(cudaFree(dCscRowInd));
    CHECK_CUDA(cudaFree(buffer));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}