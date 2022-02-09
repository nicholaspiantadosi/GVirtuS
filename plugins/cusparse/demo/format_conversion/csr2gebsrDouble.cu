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

    const int rowBlockDim = 2;
    const int colBlockDim = 2;

    int mb = (m + rowBlockDim-1)/rowBlockDim;

    double hCsrValA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    double hBsrValC_result[] = {1, 0, 0, 4, 2, 0, 3, 0, 5, 0, 0, 8, 6, 0, 7, 9};
    int hBsrRowPtrC_result[] = {0, 2, 4};
    int hBsrColIndC_result[] = {0, 1, 0, 1};

    // Device memory management
    double *dCsrValA, *dBsrValC;
    int *dCsrRowPtrA, *dCsrColIndA, *dBsrRowPtrC, *dBsrColIndC;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );

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

    // Given CSR format (csrRowPtrA, csrColIndA, csrValA) and
    // blocks of BSR format are stored in column-major order.
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
    int base, nnzb;
    int bufferSize;
    void *pBuffer;
    cusparseDcsr2gebsr_bufferSize(handle, dir, m, n,
                                  descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA,
                                  rowBlockDim, colBlockDim,
                                  &bufferSize);

    if (bufferSize == 0) {
        bufferSize = 528;
    }

    cudaMalloc((void**)&pBuffer, bufferSize);
    cudaMalloc((void**)&dBsrRowPtrC, sizeof(int) *(mb+1));
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzb;
    cusparseXcsr2gebsrNnz(handle, dir, m, n,
                          descrA, dCsrRowPtrA, dCsrColIndA,
                          descrC, dBsrRowPtrC, rowBlockDim, colBlockDim,
                          nnzTotalDevHostPtr,
                          pBuffer);
    if (NULL != nnzTotalDevHostPtr){
        nnzb = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzb, dBsrRowPtrC+mb, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, dBsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzb -= base;
    }
    cudaMalloc((void**)&dBsrColIndC, sizeof(int)*nnzb);
    cudaMalloc((void**)&dBsrValC, sizeof(double)*(rowBlockDim*colBlockDim)*nnzb);
    cusparseDcsr2gebsr(handle, dir, m, n,
                       descrA,
                       dCsrValA, dCsrRowPtrA, dCsrColIndA,
                       descrC,
                       dBsrValC, dBsrRowPtrC, dBsrColIndC,
                       rowBlockDim, colBlockDim,
                       pBuffer);

    double hBsrValC[nnzb * rowBlockDim * colBlockDim];
    int hBsrRowPtrC[mb + 1];
    int hBsrColIndC[nnzb];

    // device result check
    CHECK_CUDA( cudaMemcpy(hBsrValC, dBsrValC, nnzb * rowBlockDim * colBlockDim * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBsrRowPtrC, dBsrRowPtrC, (mb + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBsrColIndC, dBsrColIndC, nnzb * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnzb * rowBlockDim * colBlockDim; i++) {
        if((fabs(hBsrValC[i] - hBsrValC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (mb + 1); i++) {
        if((fabs(hBsrRowPtrC[i] - hBsrRowPtrC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnzb; i++) {
        if((fabs(hBsrColIndC[i] - hBsrColIndC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csr2gebsr test PASSED\n");
    else
        printf("csr2gebsr test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(dCsrValA) );
    CHECK_CUDA(cudaFree(dCsrRowPtrA) );
    CHECK_CUDA(cudaFree(dCsrColIndA) );
    CHECK_CUDA(cudaFree(dBsrValC) );
    CHECK_CUDA(cudaFree(dBsrRowPtrC) );
    CHECK_CUDA(cudaFree(dBsrColIndC) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}