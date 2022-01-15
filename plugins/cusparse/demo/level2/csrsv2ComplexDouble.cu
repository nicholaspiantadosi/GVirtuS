#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

// Interfacce che usano funcion CUSPARSE
void mat2csr(cusparseHandle_t, cusparseMatDescr_t, cuDoubleComplex *, int, int, int, cuDoubleComplex *, int *, int *);
void csrmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, cuDoubleComplex, cuDoubleComplex *, int *, int *, cuDoubleComplex *, cuDoubleComplex, cuDoubleComplex *);
void csrsv2(cusparseHandle_t, cusparseMatDescr_t, int, int, cuDoubleComplex *, int *, int *, cuDoubleComplex *, cuDoubleComplex *);

int main(int argn, char *argv[])
{
    // Variabili generiche
    int m, nnz;

    // Variabili su host
    cuDoubleComplex *matrix_host;
    cuDoubleComplex *csr_values_result;
    int *csr_columns_result, *csr_offsets_result;
    cuDoubleComplex *x_host, *y_result_csr;
    cuDoubleComplex *matrix_host_sequential;

    srand(time(0));

    m = 4;
    nnz = 5;

    // Allocazione memoria sull'host
    matrix_host = (cuDoubleComplex *)malloc((m*m)*sizeof(cuDoubleComplex));
    csr_values_result=(cuDoubleComplex *)malloc(nnz * sizeof(cuDoubleComplex));
    csr_offsets_result=(int *)malloc((m + 1) * sizeof(int));
    csr_columns_result=(int *)malloc(nnz * sizeof(int));
    x_host = (cuDoubleComplex *)malloc(m*sizeof(cuDoubleComplex));
    y_result_csr = (cuDoubleComplex *)malloc(m*sizeof(cuDoubleComplex));
    matrix_host_sequential = (cuDoubleComplex *)malloc((m*m+1)*sizeof(cuDoubleComplex));

    // Inizializzazione variabili sull'host
    initializeMatrixLowerTriangularSparseRandomZ(matrix_host, m, nnz);
    initializeArrayRandomZ(x_host, m);
    initializeArrayToZeroZ(y_result_csr, m);

    // Swap formato matrice per calcolo sequenziale su host
    swapMatrixZ(matrix_host, m, m, matrix_host_sequential);

    printf("Matrice sparsa:\n");
    stampaMatrixZ(matrix_host, m, m);
    printf("\n");

    printf("Array x:\n");
    stampaArrayZ(x_host, m);
    printf("\n");

    printf("Matrice in formato denso per calcolo sequenziale:\n");
    stampaMatrixZ1D(matrix_host_sequential, m, m);
    printf("\n");

    // Dichiarazione dell'handle per CUSPARSE
    cusparseHandle_t handle;

    // Creazione dell'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Creazione della struttura della matrice con relative proprietà e forma
    cusparseMatDescr_t descr = 0;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

    // Conversione matrice in formato CSR
    mat2csr(handle, descr, matrix_host, m, m, nnz, csr_values_result, csr_offsets_result, csr_columns_result);

    //Stampa matrice convertita in formato CSR
    printf("Matrice sparsa convertita in formato CSR\n");
    printf("\t csrValA:\t");
    stampaArrayZ(csr_values_result, nnz);
    printf("\t csrRowPtrA:\t");
    stampaArray(csr_offsets_result, (m + 1));
    printf("\t csrColIndA:\t");
    stampaArray(csr_columns_result, nnz);
    printf("\n");

    csrsv2(handle, descr, m, nnz, csr_values_result, csr_offsets_result, csr_columns_result, x_host, y_result_csr);

    printf("Vettore risultato dall'operazione \n");
    stampaArrayZ(y_result_csr, m);
    printf("\n");

    //Libera la memoria sull'host
    free(csr_values_result);
    free(csr_offsets_result);
    free(csr_columns_result);
    free(matrix_host);
    free(matrix_host_sequential);

    // Termina l'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));
}

void mat2csr(cusparseHandle_t handle, cusparseMatDescr_t descr, cuDoubleComplex * matrix, int m, int n, int nnz, cuDoubleComplex * csrValA, int * csrRowPtrA, int * csrColIndA)
{
    int nnz_total = 0;

    // Variabili su device
    cuDoubleComplex *matrix_device;
    cuDoubleComplex *csr_values_device;
    int *csr_columns_device, *csr_offsets_device;
    int *nnz_per_row;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &matrix_device, m * n * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csr_values_device, nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csr_offsets_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csr_columns_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &nnz_per_row, m * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(matrix_device, matrix, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Calcolo valori diversi da zero
    CHECK_CUSPARSE(cusparseZnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descr, matrix_device, m, nnz_per_row, &nnz_total));

    // Controllo su valori diversi da zero richiesti in input rispetto a quelli calcolati tramite cusparseSnnz()
    if (nnz != nnz_total) {
        printf("I valori diversi da zero richiesti in input sono diversi rispetto a quelli rilevati: richiesti %d valori ma sono stati rilevati %d valori diversi da zero!\n\n", nnz, nnz_total);
        exit(EXIT_FAILURE);
    }

    // Conversione matrice in formato CSR
    CHECK_CUSPARSE(cusparseZdense2csr(handle, m, n, descr, matrix_device, m, nnz_per_row, csr_values_device, csr_offsets_device, csr_columns_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(csrValA, csr_values_device, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csrRowPtrA, csr_offsets_device, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csrColIndA, csr_columns_device, nnz * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(csr_values_device));
    CHECK_CUDA(cudaFree(csr_offsets_device));
    CHECK_CUDA(cudaFree(csr_columns_device));
    CHECK_CUDA(cudaFree(matrix_device));
    CHECK_CUDA(cudaFree(nnz_per_row));
}

// Suppose that L is m x m sparse matrix represented by CSR format,
// L is lower triangular with unit diagonal.
// Assumption:
// - dimension of matrix L is m,
// - matrix L has nnz number zero elements,
// - handle is already created by cusparseCreate(),
// - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of L on device memory,
// - d_x is right hand side vector on device memory,
// - d_y is solution vector on device memory.
void csrsv2(cusparseHandle_t handle, cusparseMatDescr_t descr, int m, int nnz, cuDoubleComplex * csrVal, int * csrRowPtr, int * csrColInd, cuDoubleComplex * x, cuDoubleComplex * y) {

    cuDoubleComplex *d_csrVal;
    int *d_csrRowPtr, *d_csrColInd;
    cuDoubleComplex *d_x, *d_y;

    CHECK_CUDA(cudaMalloc((void**) &d_csrVal, nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &d_csrRowPtr, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &d_csrColInd, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &d_x, m * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &d_y, m * sizeof(cuDoubleComplex)));

    CHECK_CUDA(cudaMemcpy(d_x, x, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, y, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrVal, csrVal, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, csrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColInd, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

    csrsv2Info_t info = 0;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0);
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
    //   (L may not have all diagonal elements.)
    cusparseCreateMatDescr(&descr);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatFillMode(descr, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr, CUSPARSE_DIAG_TYPE_UNIT);

    // step 2: create a empty info structure
    cusparseCreateCsrsv2Info(&info);

    // step 3: query how much memory used in csrsv2, and allocate the buffer
    cusparseZcsrsv2_bufferSize(handle, trans, m, nnz, descr,
                               d_csrVal, d_csrRowPtr, d_csrColInd, info, &pBufferSize);
    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis
    cusparseZcsrsv2_analysis(handle, trans, m, nnz, descr,
                             d_csrVal, d_csrRowPtr, d_csrColInd,
                             info, policy, pBuffer);
    // L has unit diagonal, so no structural zero is reported.
    cusparseStatus_t status = cusparseXcsrsv2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // step 5: solve L*y = x
    cusparseZcsrsv2_solve(handle, trans, m, nnz, &alpha, descr,
                          d_csrVal, d_csrRowPtr, d_csrColInd, info,
                          d_x, d_y, policy, pBuffer);
    // L has unit diagonal, so no numerical zero is reported.
    status = cusparseXcsrsv2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    CHECK_CUDA(cudaMemcpy(y, d_y, m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    // step 6: free resources
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    cudaFree(pBuffer);
    cusparseDestroyCsrsv2Info(info);
}