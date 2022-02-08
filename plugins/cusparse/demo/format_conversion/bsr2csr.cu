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

    const int blockDim = 2;
    const int mb = 2;

    float hBsrValA[] = {1, 0, 0, 4, 2, 0, 3, 0, 5, 0, 0, 8, 6, 0, 7, 9};
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    int m = mb * blockDim;
    int nnzb = 4; // number of blocks
    int nnz  = nnzb * blockDim * blockDim; // number of elements

    float hCsrValC[nnz];
    int hCsrRowPtrC[m+1];
    int hCsrColIndC[nnz];

    float hCsrValC_result[] = {1.000000, 0.000000, 2.000000, 3.000000, 0.000000, 4.000000, 0.000000, 0.000000, 5.000000, 0.000000, 6.000000, 7.000000, 0.000000, 8.000000, 0.000000, 9.000000};
    int hCsrRowPtrC_result[] = {0, 4, 8, 12, 16};
    int hCsrColIndC_result[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    // Device memory management

    float *dBsrValA, *dCsrValC;
    int *dBsrRowPtrA, *dBsrColIndA, *dCsrRowPtrC, *dCsrColIndC;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (blockDim * blockDim) * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrValC, nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrC, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndC, nnz * sizeof(int)));

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (blockDim * blockDim) * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValC, hCsrValC, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrC, hCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndC, hCsrColIndC, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseMatDescr_t descrC = 0;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL );

    // Given BSR format (bsrRowPtrA, bsrcolIndA, bsrValA) and
    // blocks of BSR format are stored in column-major order.
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

    cusparseSbsr2csr(handle, dir, mb, mb,
                     descrA,
                     dBsrValA, dBsrRowPtrA, dBsrColIndA,
                     blockDim,
                     descrC,
                     dCsrValC, dCsrRowPtrC, dCsrColIndC);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrValC[i] - hCsrValC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
        if(hCsrColIndC[i] - hCsrColIndC_result[i] > 0) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (m + 1); i++) {
        if((hCsrRowPtrC[i] - hCsrRowPtrC_result[i] > 0)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("bsr2csr test PASSED\n");
    else
        printf("bsr2csr test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(dCsrValC));
    CHECK_CUDA(cudaFree(dCsrRowPtrC));
    CHECK_CUDA(cudaFree(dCsrColIndC));
    CHECK_CUDA(cudaFree(dBsrValA));
    CHECK_CUDA(cudaFree(dBsrRowPtrA));
    CHECK_CUDA(cudaFree(dBsrColIndA));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}