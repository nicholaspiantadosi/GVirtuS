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
    const int rowBlockDim = 2;
    const int colBlockDim = 2;
    const int mb = 2;
    const int nb = 2;
    const int nnzb = 4;

    double hBsrValA[] = {1, 0, 0, 4, 2, 0, 3, 0, 5, 0, 0, 8, 6, 0, 7, 4};
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    double hBscVal[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int hBscRowInd[] = {0, 0, 0, 0};
    int hBscColPtr[] = {0, 0, 0};

    double hBscVal_result[] = {1, 0, 0, 4, 5, 0, 0, 8, 2, 0, 3, 0, 6, 0, 7, 4};
    int hBscRowInd_result[] = {0, 1, 0, 1};
    int hBscColPtr_result[] = {0, 2, 4};

    // Device memory management
    double *dBsrValA, *dBscVal;
    int *dBsrRowPtrA, *dBsrColIndA, *dBscRowInd, *dBscColPtr;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (rowBlockDim * colBlockDim) * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBscVal,  nnzb * (rowBlockDim * colBlockDim) * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dBscRowInd, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBscColPtr, (nb + 1) * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (rowBlockDim * colBlockDim) * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBscVal, hBscVal, nnzb * (rowBlockDim * colBlockDim) * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBscRowInd, hBscRowInd, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBscColPtr, hBscColPtr, (nb + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    int pBufferSize;
    void *pBuffer = 0;

    cusparseDgebsr2gebsc_bufferSize(handle, mb, nb, nnzb, dBsrValA, dBsrRowPtrA, dBsrColIndA, rowBlockDim, colBlockDim, &pBufferSize);

    cudaMalloc((void**)&pBuffer, pBufferSize);

    cusparseDgebsr2gebsc(handle, mb, nb, nnzb, dBsrValA, dBsrRowPtrA, dBsrColIndA, rowBlockDim,
                         colBlockDim, dBscVal, dBscRowInd, dBscColPtr, CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hBscVal, dBscVal, nnzb * (rowBlockDim * colBlockDim) * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBscRowInd, dBscRowInd, nnzb * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBscColPtr, dBscColPtr, (nb + 1) * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnzb * (rowBlockDim * colBlockDim); i++) {
        if((fabs(hBscVal[i] - hBscVal_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnzb; i++) {
        if((fabs(hBscRowInd[i] - hBscRowInd_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (nb + 1); i++) {
        if((fabs(hBscColPtr[i] - hBscColPtr_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("gebsr2gebsc test PASSED\n");
    else
        printf("gebsr2gebsc test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(dBsrValA) );
    CHECK_CUDA(cudaFree(dBsrRowPtrA) );
    CHECK_CUDA(cudaFree(dBsrColIndA) );
    CHECK_CUDA(cudaFree(dBscVal) );
    CHECK_CUDA(cudaFree(dBscRowInd) );
    CHECK_CUDA(cudaFree(dBscColPtr) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}