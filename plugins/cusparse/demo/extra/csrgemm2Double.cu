#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[]) {

    // Host problem definition

    int m = 4;
    int n = 4;
    int k = 4;
    double alpha = 1;
    double beta = 1;

    int nnzA = 9;
    double hCsrValA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    int nnzB = 8;
    double hCsrValB[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int hCsrRowPtrB[] = {0, 3, 4, 6, 8};
    int hCsrColIndB[] = {0, 1, 3, 3, 2, 3, 2, 3};

    int nnzD = 7;
    double hCsrValD[] = {1, 2, 3, 4, 5, 6, 7};
    int hCsrRowPtrD[] = {0, 1, 4, 5, 7};
    int hCsrColIndD[] = {0, 0, 2, 3, 2, 2, 3};

    double hCsrValC_result[] = {2, 2, 31, 39, 2, 3, 20, 5, 10, 84, 107, 69, 111};
    int hCsrRowPtrC_result[] = {0, 4, 7, 11, 13};
    int hCsrColIndC_result[] = {0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3, 2, 3};

    // Device memory management

    double *dCsrValA, *dCsrValB, *dCsrValC, *dCsrValD;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrB, *dCsrColIndB, *dCsrRowPtrC, *dCsrColIndC, *dCsrRowPtrD, *dCsrColIndD;

    dCsrValC = NULL;
    dCsrRowPtrC = NULL;
    dCsrColIndC = NULL;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA, nnzA * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnzA * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrValB, nnzB * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrB, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndB, nnzB * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrValD, nnzD * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrD, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndD, nnzD * sizeof(int)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValB, hCsrValB, nnzB * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrB, hCsrRowPtrB, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndB, hCsrColIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValD, hCsrValD, nnzD * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrD, hCsrRowPtrD, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndD, hCsrColIndD, nnzD * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseMatDescr_t descrB = 0;
    cusparseCreateMatDescr(&descrB);
    cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseMatDescr_t descrC = 0;
    cusparseCreateMatDescr(&descrC);
    cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseMatDescr_t descrD = 0;
    cusparseCreateMatDescr(&descrD);
    cusparseSetMatType(descrD, CUSPARSE_MATRIX_TYPE_GENERAL );

    int baseC, nnzC;
    csrgemm2Info_t info = NULL;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t bufferSize;
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = &nnzC;

    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    cusparseCreateCsrgemm2Info(&info);

    cusparseDcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha,
                                    descrA, nnzA, dCsrRowPtrA, dCsrColIndA,
                                    descrB, nnzB, dCsrRowPtrB, dCsrColIndB,
                                    &beta,
                                    descrD, nnzD, dCsrRowPtrD, dCsrColIndD,
                                    info,
                                    &bufferSize);

    cudaMalloc(&buffer, bufferSize);
    cudaMalloc((void**)&dCsrRowPtrC, sizeof(int)*(m+1));

    cusparseXcsrgemm2Nnz(handle, m, n, k,
                         descrA, nnzA, dCsrRowPtrA, dCsrColIndA,
                         descrB, nnzB, dCsrRowPtrB, dCsrColIndB,
                         descrD, nnzD, dCsrRowPtrD, dCsrColIndD,
                         descrC, dCsrRowPtrC, nnzTotalDevHostPtr,
                         info, buffer );

    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, dCsrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, dCsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }

    cudaMalloc((void**)&dCsrColIndC, sizeof(int)*nnzC);
    cudaMalloc((void**)&dCsrValC, sizeof(double)*nnzC);

    cusparseDcsrgemm2(handle, m, n, k, &alpha,
                      descrA, nnzA, dCsrValA, dCsrRowPtrA, dCsrColIndA,
                      descrB, nnzB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
                      &beta,
                      descrD, nnzD, dCsrValD, dCsrRowPtrD, dCsrColIndD,
                      descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC,
                      info, buffer);

    cusparseDestroyCsrgemm2Info(info);

    // device result check

    double hCsrValC[nnzC];
    int hCsrRowPtrC[m+1];
    int hCsrColIndC[nnzC];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzC * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzC * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;

    for (int i = 0; i < nnzC; i++) {
        if(hCsrColIndC[i] != hCsrColIndC_result[i]) {
            correct = 0;
            break;
        }
    }

    for (int i = 0; i < (m + 1); i++) {
        if(hCsrRowPtrC[i] != hCsrRowPtrC_result[i]) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csrgeam2 test PASSED\n");
    else
        printf("csrgeam2 test FAILED: wrong result\n");

    // destroy

    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation

    CHECK_CUDA( cudaFree(dCsrValA) );
    CHECK_CUDA( cudaFree(dCsrRowPtrA) );
    CHECK_CUDA( cudaFree(dCsrColIndA) );
    CHECK_CUDA( cudaFree(dCsrValB) );
    CHECK_CUDA( cudaFree(dCsrRowPtrB) );
    CHECK_CUDA( cudaFree(dCsrColIndB) );
    CHECK_CUDA( cudaFree(dCsrValC) );
    CHECK_CUDA( cudaFree(dCsrRowPtrC) );
    CHECK_CUDA( cudaFree(dCsrColIndC) );
    CHECK_CUDA( cudaFree(dCsrValD) );
    CHECK_CUDA( cudaFree(dCsrRowPtrD) );
    CHECK_CUDA( cudaFree(dCsrColIndD) );

    return EXIT_SUCCESS;
}