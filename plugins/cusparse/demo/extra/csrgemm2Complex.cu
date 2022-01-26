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
    cuComplex alpha = make_cuComplex(1, 0);
    cuComplex beta = make_cuComplex(1, 0);

    int nnzA = 9;
    cuComplex hCsrValA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(5, 0), make_cuComplex(6, 0), make_cuComplex(7, 0), make_cuComplex(8, 0), make_cuComplex(9, 0)};
    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    int nnzB = 8;
    cuComplex hCsrValB[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(5, 0), make_cuComplex(6, 0), make_cuComplex(7, 0), make_cuComplex(8, 0)};
    int hCsrRowPtrB[] = {0, 3, 4, 6, 8};
    int hCsrColIndB[] = {0, 1, 3, 3, 2, 3, 2, 3};

    int nnzD = 7;
    cuComplex hCsrValD[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(5, 0), make_cuComplex(6, 0), make_cuComplex(7, 0)};
    int hCsrRowPtrD[] = {0, 1, 4, 5, 7};
    int hCsrColIndD[] = {0, 0, 2, 3, 2, 2, 3};

    cuComplex hCsrValC_result[] = {make_cuComplex(2, 0), make_cuComplex(2, 0), make_cuComplex(31, 0), make_cuComplex(39, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(20, 0), make_cuComplex(5, 0), make_cuComplex(10, 0), make_cuComplex(84, 0), make_cuComplex(107, 0), make_cuComplex(69, 0), make_cuComplex(111, 0)};
    int hCsrRowPtrC_result[] = {0, 4, 7, 11, 13};
    int hCsrColIndC_result[] = {0, 1, 2, 3, 0, 2, 3, 0, 1, 2, 3, 2, 3};

    // Device memory management

    cuComplex *dCsrValA, *dCsrValB, *dCsrValC, *dCsrValD;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrB, *dCsrColIndB, *dCsrRowPtrC, *dCsrColIndC, *dCsrRowPtrD, *dCsrColIndD;

    dCsrValC = NULL;
    dCsrRowPtrC = NULL;
    dCsrColIndC = NULL;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA, nnzA * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnzA * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrValB, nnzB * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrB, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndB, nnzB * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrValD, nnzD * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrD, (m + 1) * sizeof(int)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndD, nnzD * sizeof(int)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValB, hCsrValB, nnzB * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrB, hCsrRowPtrB, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndB, hCsrColIndB, nnzB * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrValD, hCsrValD, nnzD * sizeof(cuComplex), cudaMemcpyHostToDevice) );
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

    cusparseCcsrgemm2_bufferSizeExt(handle, m, n, k, &alpha,
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
    cudaMalloc((void**)&dCsrValC, sizeof(cuComplex)*nnzC);

    cusparseCcsrgemm2(handle, m, n, k, &alpha,
                      descrA, nnzA, dCsrValA, dCsrRowPtrA, dCsrColIndA,
                      descrB, nnzB, dCsrValB, dCsrRowPtrB, dCsrColIndB,
                      &beta,
                      descrD, nnzD, dCsrValD, dCsrRowPtrD, dCsrColIndD,
                      descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC,
                      info, buffer);

    cusparseDestroyCsrgemm2Info(info);

    // device result check

    cuComplex hCsrValC[nnzC];
    int hCsrRowPtrC[m+1];
    int hCsrColIndC[nnzC];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzC * sizeof(cuComplex), cudaMemcpyDeviceToHost) );
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