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
    int n = 2;
    int nnz = 4;

    int hCscColPtr[] = {0, 2, 4};
    int hCscRowInd[] = {1, 0, 2, 0};

    int hCscRowInd_result[] = {0, 1, 0, 2};

    // Device memory management
    int *dCscColPtr, *dCscRowInd, *dp;

    CHECK_CUDA( cudaMalloc((void**) &dCscColPtr, (n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscRowInd, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dp, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCscColPtr, hCscColPtr, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscRowInd, hCscRowInd, nnz * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    size_t pBufferSizeInBytes = 0;
    void *pBuffer = NULL;

    CHECK_CUSPARSE(cusparseXcscsort_bufferSizeExt(handle, m, n, nnz, dCscColPtr, dCscRowInd, &pBufferSizeInBytes));

    if (pBufferSizeInBytes == 0) {
        pBufferSizeInBytes = 1280;
    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSizeInBytes * sizeof(char)));

    CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, nnz, dp));

    CHECK_CUSPARSE(cusparseXcscsort(handle, m, n, nnz, descrA, dCscColPtr, dCscRowInd, dp, pBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hCscRowInd, dCscRowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCscRowInd[i] - hCscRowInd_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("cscsort test PASSED\n");
    else
        printf("cscsort test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUDA(cudaFree(dCscColPtr) );
    CHECK_CUDA(cudaFree(dCscRowInd) );
    CHECK_CUDA(cudaFree(dp) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}