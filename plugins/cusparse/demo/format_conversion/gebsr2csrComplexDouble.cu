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
    int m = mb * rowBlockDim;
    int nnz  = nnzb * rowBlockDim * colBlockDim; // number of elements

    cuDoubleComplex hBsrValA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(4, 0),
                            make_cuDoubleComplex(2, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(0, 0),
                            make_cuDoubleComplex(5, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(8, 0),
                            make_cuDoubleComplex(6, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(9, 0)};;
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    cuDoubleComplex hCsrValC[nnz];
    int hCsrRowPtrC[m + 1];
    int hCsrColIndC[nnz];

    cuDoubleComplex hCsrValC_result[] = {make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(2.000000, 0), make_cuDoubleComplex(3.000000, 0),
                                   make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(4.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.000000, 0),
                                   make_cuDoubleComplex(5.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(6.000000, 0), make_cuDoubleComplex(7.000000, 0),
                                   make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(8.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(9.000000, 0)};
    int hCsrRowPtrC_result[] = {0, 4, 8, 12, 16};
    int hCsrColIndC_result[] = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

    // Device memory management
    cuDoubleComplex *dBsrValA, *dCsrValC;
    int *dBsrRowPtrA, *dBsrColIndA, *dCsrRowPtrC, *dCsrColIndC;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (rowBlockDim * colBlockDim) * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrValC,  nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrC, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndC, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (rowBlockDim * colBlockDim) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );

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

    // Given general BSR format (bsrRowPtrA, bsrColIndA, bsrValA) and
    // blocks of BSR format are stored in column-major order.
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

    cusparseZgebsr2csr(handle, dir, mb, nb,
                       descrA,
                       dBsrValA, dBsrRowPtrA, dBsrColIndA,
                       rowBlockDim, colBlockDim,
                       descrC,
                       dCsrValC, dCsrRowPtrC, dCsrColIndC);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrValC[i].x - hCsrValC_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
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
        printf("gebsr2csr test PASSED\n");
    else
        printf("gebsr2csr test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dBsrValA) );
    CHECK_CUDA(cudaFree(dBsrRowPtrA) );
    CHECK_CUDA(cudaFree(dBsrColIndA) );
    CHECK_CUDA(cudaFree(dCsrValC) );
    CHECK_CUDA(cudaFree(dCsrRowPtrC) );
    CHECK_CUDA(cudaFree(dCsrColIndC) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}