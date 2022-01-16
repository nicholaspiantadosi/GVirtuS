#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[]) {

    // Host problem definition

    int m = 4;
    int n = 5;
    float alpha = 1;
    float beta = 1;

    int nnzA = 9;
    float hCsrValA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    int nnzB = 8;
    float hCsrValB[] = {1, 2, 3, 4, 5, 6, 7, 8};
    int hCsrRowPtrB[] = {0, 3, 4, 6, 8};
    int hCsrColIndB[] = {0, 1, 3, 3, 2, 3, 2, 3};

    float hCsrValC_result[] = {2, 2, 2, 6, 4, 4, 5, 11, 13, 8, 7, 17};
    int hCsrRowPtrC_result[] = {0, 4, 6, 9, 12};
    int hCsrColIndC_result[] = {0, 1, 2, 3, 1, 3, 0, 2, 3, 1, 2, 3};

    // Device memory management

    float *dCsrValA, *dCsrValB, *dCsrValC;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrB, *dCsrColIndB, *dCsrRowPtrC, *dCsrColIndC;

    dCsrValC = NULL;
    dCsrRowPtrC = NULL;
    dCsrColIndC = NULL;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA, nnzA * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnzA * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrValB, nnzB * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrB, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndB, nnzB * sizeof(int)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValB, hCsrValB, nnzB * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrB, hCsrRowPtrB, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndB, hCsrColIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice) );

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

    int baseC, nnzC;
    /* alpha, nnzTotalDevHostPtr points to host memory */
    size_t bufferSizeInBytes;
    char *buffer = NULL;
    int *nnzTotalDevHostPtr = &nnzC;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    cudaMalloc((void**)&dCsrRowPtrC, sizeof(int)*(m+1));
    cusparseScsrgeam2_bufferSizeExt(handle, m, n, &alpha,
                                    descrA, nnzA, dCsrValA, dCsrRowPtrA, dCsrColIndA, &beta,
                                    descrB, nnzB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
                                    descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC,
                                    &bufferSizeInBytes);
    cudaMalloc((void**)&buffer, sizeof(char)*bufferSizeInBytes);
    cusparseXcsrgeam2Nnz(handle, m, n,
                         descrA, nnzA, dCsrRowPtrA, dCsrColIndA,
                         descrB, nnzB, dCsrRowPtrB, dCsrColIndB,
                         descrC, dCsrRowPtrC, nnzTotalDevHostPtr,
                         buffer);
    if (NULL != nnzTotalDevHostPtr){
        nnzC = *nnzTotalDevHostPtr;
    }else{
        cudaMemcpy(&nnzC, dCsrRowPtrC+m, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&baseC, dCsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
        nnzC -= baseC;
    }
    cudaMalloc((void**)&dCsrColIndC, sizeof(int)*nnzC);
    cudaMalloc((void**)&dCsrValC, sizeof(float)*nnzC);
    cusparseScsrgeam2(handle, m, n, &alpha, descrA, nnzA, dCsrValA, dCsrRowPtrA, dCsrColIndA, &beta, descrB, nnzB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
                      descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC, buffer);

    // device result check

    float hCsrValC[nnzC];
    int hCsrRowPtrC[m+1];
    int hCsrColIndC[nnzC];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzC * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzC * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;

    for (int i = 0; i < nnzC; i++) {
        if((fabs(hCsrValC[i] - hCsrValC_result[i]) > 0.000001) ||
                hCsrColIndC[i] != hCsrColIndC_result[i]) {
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

    return EXIT_SUCCESS;
}