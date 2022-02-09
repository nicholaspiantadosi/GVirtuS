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
    const int rowBlockDimA = 2;
    const int colBlockDimA = 2;
    const int mb = 2;
    const int nb = 2;
    const int nnzb = 4;
    const int rowBlockDimC = 2;
    const int colBlockDimC = 2;
    int m = mb * rowBlockDimA;
    int mc = (m + rowBlockDimC - 1)/rowBlockDimC;

    cuComplex hBsrValA[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0), make_cuComplex(0, 0),
                            make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(8, 0), make_cuComplex(6, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(9, 0)};
    int hBsrRowPtrA[] = {0, 2, 4};
    int hBsrColIndA[] = {0, 1, 0, 1};

    cuComplex hBsrValC_result[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0), make_cuComplex(0, 0),
                                   make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(8, 0), make_cuComplex(6, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(9, 0)};
    int hBsrRowPtrC_result[] = {0, 2, 4};
    int hBsrColIndC_result[] = {0, 1, 0, 1};

    // Device memory management
    cuComplex *dBsrValA, *dBsrValC;
    int *dBsrRowPtrA, *dBsrColIndA, *dBsrRowPtrC, *dBsrColIndC;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (rowBlockDimA * colBlockDimA) * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (rowBlockDimA * colBlockDimA) * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

    int base, nnzc;
    int bufferSize;
    void *pBuffer = 0;

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseMatDescr_t descrC = 0;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseCgebsr2gebsr_bufferSize(handle, dir, mb, nb, nnzb,
                                    descrA, hBsrValA, hBsrRowPtrA, hBsrColIndA,
                                    rowBlockDimA, colBlockDimA,
                                    rowBlockDimC, colBlockDimC,
                                    &bufferSize);
    if(bufferSize == 0) {
        bufferSize = 912;
    }
    cudaMalloc((void**)&pBuffer, bufferSize);
    cudaMalloc((void**)&dBsrRowPtrC, sizeof(int)*(mc+1));
    // nnzTotalDevHostPtr points to host memory
    int *nnzTotalDevHostPtr = &nnzc;
    cusparseXgebsr2gebsrNnz(handle, dir, mb, nb, nnzb,
                            descrA, dBsrRowPtrA, dBsrColIndA,
                            rowBlockDimA, colBlockDimA,
                            descrC, dBsrRowPtrC,
                            rowBlockDimC, colBlockDimC,
                            nnzTotalDevHostPtr,
                            pBuffer);
    if (NULL != nnzTotalDevHostPtr){
        nnzc = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzc, dBsrRowPtrC + mc, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base, dBsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzc -= base;
    }
    cudaMalloc((void**)&dBsrColIndC, sizeof(int)*nnzc);
    cudaMalloc((void**)&dBsrValC, sizeof(cuComplex)*(rowBlockDimC*colBlockDimC)*nnzc);
    cusparseCgebsr2gebsr(handle, dir, mb, nb, nnzb,
                         descrA, dBsrValA, dBsrRowPtrA, dBsrColIndA,
                         rowBlockDimA, colBlockDimA,
                         descrC, dBsrValC, dBsrRowPtrC, dBsrColIndC,
                         rowBlockDimC, colBlockDimC,
                         pBuffer);

    // device result check
    cuComplex hBsrValC[nnzc * rowBlockDimC * colBlockDimC];
    int hBsrRowPtrC[mc + 1];
    int hBsrColIndC[nnzc];

    CHECK_CUDA( cudaMemcpy(hBsrValC, dBsrValC, nnzc * (rowBlockDimC * colBlockDimC) * sizeof(cuComplex), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBsrRowPtrC, dBsrRowPtrC, (mc + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hBsrColIndC, dBsrColIndC, nnzc * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnzc * (rowBlockDimC * colBlockDimC); i++) {
        if((fabs(hBsrValC[i].x - hBsrValC_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (mc + 1); i++) {
        if((fabs(hBsrRowPtrC[i] - hBsrRowPtrC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnzc; i++) {
        if((fabs(hBsrColIndC[i] - hBsrColIndC_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("gebsr2gebsr test PASSED\n");
    else
        printf("gebsr2gebsr test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(dBsrValA) );
    CHECK_CUDA(cudaFree(dBsrRowPtrA) );
    CHECK_CUDA(cudaFree(dBsrColIndA) );
    CHECK_CUDA(cudaFree(dBsrValC) );
    CHECK_CUDA(cudaFree(dBsrRowPtrC) );
    CHECK_CUDA(cudaFree(dBsrColIndC) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}