#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

// Interfacce che usano funcion CUSPARSE
void mat2csr(cusparseHandle_t, cusparseMatDescr_t, cuDoubleComplex *, int, int, int, cuDoubleComplex *, int *, int *);
void csrmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, cuDoubleComplex, cuDoubleComplex *, int *, int *, cuDoubleComplex *, cuDoubleComplex, cuDoubleComplex *);
void bsrnnz(cusparseHandle_t, cusparseMatDescr_t, int *, int *, int, int, int, int, int, int, int *, int &);
void csr2bsr(cusparseHandle_t, cusparseMatDescr_t, cuDoubleComplex *, int *, int *, int, int, int, int, int , int , int, int *,  cuDoubleComplex *, int *);
void bsrsv2(cusparseHandle_t, cusparseMatDescr_t, int, int, int, cuDoubleComplex *, int *, int *, int, cuDoubleComplex *, cuDoubleComplex *);

int main(int argn, char *argv[])
{
    // Variabili generiche
    int m, nnz, blockDim, mb=0, nnzb=0;

    // Variabili su host
    cuDoubleComplex *matrix_host;
    cuDoubleComplex *csr_values_result;
    int *csr_columns_result, *csr_offsets_result;
    cuDoubleComplex *bsr_values_result;
    int *bsr_columns_result, *bsr_offsets_result;
    cuDoubleComplex *x_host, *y_result_bsr;
    cuDoubleComplex *matrix_host_sequential;

    srand(time(0));

    m = 4;
    nnz = 5;
    blockDim = 2;

    // Allocazione memoria sull'host
    matrix_host = (cuDoubleComplex *)malloc((m*m)*sizeof(cuDoubleComplex));
    csr_values_result=(cuDoubleComplex *)malloc(nnz * sizeof(cuDoubleComplex));
    csr_offsets_result=(int *)malloc((m + 1) * sizeof(int));
    csr_columns_result=(int *)malloc(nnz * sizeof(int));
    x_host = (cuDoubleComplex *)malloc(m*sizeof(cuDoubleComplex));
    y_result_bsr = (cuDoubleComplex *)malloc(m*sizeof(cuDoubleComplex));
    matrix_host_sequential = (cuDoubleComplex *)malloc((m*m+1)*sizeof(cuDoubleComplex));

    // Inizializzazione variabili sull'host
    initializeMatrixLowerTriangularSparseRandomZ(matrix_host, m, nnz);
    initializeArrayRandomZ(x_host, m);
    initializeArrayToZeroZ(y_result_bsr, m);

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

    // Calcolo mb a partire da blockDim
    mb = (m + blockDim - 1) / blockDim;

    // Allocazione variabili su host
    bsr_offsets_result=(int *)malloc((mb + 1) * sizeof(int));

    // Calcolo bsrRowPtrC e blocchi diversi da zero
    bsrnnz(handle, descr, csr_offsets_result, csr_columns_result, m, m, nnz, blockDim, mb, mb, bsr_offsets_result, nnzb);

    // Allocazione variabili su host sulla base del numero di blocchi diversi da zero
    bsr_columns_result=(int *)malloc(nnzb * sizeof(int));
    bsr_values_result=(cuDoubleComplex *)malloc((blockDim * blockDim) * nnzb * sizeof(cuDoubleComplex));

    // Conversione da formato CSR a BSR
    csr2bsr(handle, descr, csr_values_result, csr_offsets_result, csr_columns_result, m, m, nnz, blockDim, mb, mb, nnzb, bsr_offsets_result, bsr_values_result, bsr_columns_result);

    // Stampa matrice convertita in formato BSR
    printf("Matrice sparsa convertita in formato BSR\n");
    printf("\t bsrValC:\t");
    stampaArrayZ(bsr_values_result, (blockDim * blockDim) * nnzb);
    printf("\t bsrRowPtrC:\t");
    stampaArray(bsr_offsets_result, (mb + 1));
    printf("\t bsrColIndC:\t");
    stampaArray(bsr_columns_result, nnzb);
    printf("\n");

    bsrsv2(handle, descr, m, mb, nnzb, bsr_values_result, bsr_offsets_result, bsr_columns_result, blockDim, x_host, y_result_bsr);

    printf("Vettore risultato dall'operazione \n");
    stampaArrayZ(y_result_bsr, m);
    printf("\n");

    //Libera la memoria sull'host
    free(csr_values_result);
    free(csr_offsets_result);
    free(csr_columns_result);
    free(bsr_values_result);
    free(bsr_offsets_result);
    free(bsr_columns_result);
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

    // Controllo su valori diversi da zero richiesti in input rispetto a quelli calcolati tramite cusparseZnnz()
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

void bsrnnz(cusparseHandle_t handle, cusparseMatDescr_t descr, int * csrRowPtrA, int * csrColIndA, int m, int n, int nnz, int blockDim, int mb, int nb, int * bsrRowPtrC, int &nnzb)
{
    // Variabili su host
    int base;
    int *nnzTotalBsr = &nnzb;

    // Variabili su device
    int *bsrRowPtrC_device;
    int *csrRowPtrA_device, *csrColIndA_device;

    // Allocazione variabili su device
    CHECK_CUDA(cudaMalloc((void**) &csrRowPtrA_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csrColIndA_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrRowPtrC_device, (mb + 1) * sizeof(int)));

    // Copia variabili da host a device
    CHECK_CUDA(cudaMemcpy(csrRowPtrA_device, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrColIndA_device, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Calcolo del numero dei blocchi diversi da zero per il formato BSR
    CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrRowPtrA_device, csrColIndA_device, blockDim, descr, bsrRowPtrC_device, nnzTotalBsr));

    // Controllo sul valore dei blocchi calcolato, se null lo calcolo sulla base degli indici dei blocchi e il numero dei blocchi per riga
    if (NULL != nnzTotalBsr)
    {
        nnzb = *nnzTotalBsr;
    }
    else
    {
        CHECK_CUDA(cudaMemcpy(&nnzb, bsrRowPtrC_device + mb, sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(&base, bsrRowPtrC_device, sizeof(int), cudaMemcpyDeviceToHost));
        nnzb -= base;
    }

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(bsrRowPtrC, bsrRowPtrC_device, (mb + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(bsrRowPtrC_device));
    CHECK_CUDA(cudaFree(csrRowPtrA_device));
    CHECK_CUDA(cudaFree(csrColIndA_device));

}

void csr2bsr(cusparseHandle_t handle, cusparseMatDescr_t descr, cuDoubleComplex * csrValA, int * csrRowPtrA, int * csrColIndA, int m, int n, int nnz, int blockDim, int mb, int nb, int nnzb, int * bsrRowPtrC, cuDoubleComplex * bsrValC, int * bsrColIndC)
{
    // Variabili su device
    cuDoubleComplex *bsrValC_device;
    int *bsrRowPtrC_device, *bsrColIndC_device;
    cuDoubleComplex *csrValA_device;
    int *csrRowPtrA_device, *csrColIndA_device;

    // Allocazione variabili su device sulla base del numero di blocchi diversi da zero
    CHECK_CUDA(cudaMalloc((void**) &csrValA_device, nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csrRowPtrA_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csrColIndA_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrColIndC_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**)&bsrRowPtrC_device, (mb + 1) * sizeof(int)));

    // Copia da host a device
    CHECK_CUDA(cudaMemcpy(csrValA_device, csrValA, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrRowPtrA_device, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrColIndA_device, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtrC_device, bsrRowPtrC, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Conversione da CSR a BSR
    CHECK_CUSPARSE(cusparseZcsr2bsr(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrValA_device, csrRowPtrA_device, csrColIndA_device, blockDim, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(bsrValC, bsrValC_device, ((blockDim * blockDim) * nnzb) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bsrColIndC, bsrColIndC_device, nnzb * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(bsrValC_device));
    CHECK_CUDA(cudaFree(bsrColIndC_device));
    //CHECK_CUDA(cudaFree(bsrRowPtrC_device));
    CHECK_CUDA(cudaFree(csrValA_device));
    CHECK_CUDA(cudaFree(csrRowPtrA_device));
    CHECK_CUDA(cudaFree(csrColIndA_device));

}

// Suppose that L is m x m sparse matrix represented by BSR format,
// The number of block rows/columns is mb, and
// the number of nonzero blocks is nnzb.
// L is lower triangular with unit diagonal.
// Assumption:
// - dimension of matrix L is m(=mb*blockDim),
// - matrix L has nnz(=nnzb*blockDim*blockDim) nonzero elements,
// - handle is already created by cusparseCreate(),
// - (d_bsrRowPtr, d_bsrColInd, d_bsrVal) is BSR of L on device memory,
// - d_x is right hand side vector on device memory.
// - d_y is solution vector on device memory.
// - d_x and d_y are of size m.
void bsrsv2(cusparseHandle_t handle, cusparseMatDescr_t descr, int m, int mb, int nnzb, cuDoubleComplex * bsrVal, int * bsrRowPtr, int * bsrColInd, int blockDim, cuDoubleComplex * x, cuDoubleComplex * y) {

    cuDoubleComplex *bsrVal_device;
    int *bsrRowPtr_device, *bsrColInd_device;
    cuDoubleComplex *x_device, *y_device;

    CHECK_CUDA(cudaMalloc((void**) &bsrVal_device, (blockDim * blockDim) * nnzb * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &bsrRowPtr_device, (mb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &bsrColInd_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &x_device, (mb*blockDim) * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &y_device, (mb*blockDim) * sizeof(cuDoubleComplex)));

    CHECK_CUDA(cudaMemcpy(x_device, x, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrVal_device, bsrVal, (blockDim * blockDim) * nnzb * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtr_device, bsrRowPtr, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrColInd_device, bsrColInd, nnzb * sizeof(int), cudaMemcpyHostToDevice));

    bsrsv2Info_t info = 0;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cuDoubleComplex alpha = make_cuDoubleComplex(1., 0);
    const cusparseSolvePolicy_t policy = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

// step 1: create a descriptor which contains
// - matrix L is base-1
// - matrix L is lower triangular
// - matrix L has unit diagonal, specified by parameter CUSPARSE_DIAG_TYPE_UNIT
//   (L may not have all diagonal elements.)

// step 2: create a empty info structure
    cusparseCreateBsrsv2Info(&info);

// step 3: query how much memory used in bsrsv2, and allocate the buffer
    cusparseZbsrsv2_bufferSize(handle, dir, trans, mb, nnzb, descr,
                               bsrVal_device, bsrRowPtr_device, bsrColInd_device, blockDim, info, &pBufferSize);

// pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

// step 4: perform analysis
    cusparseZbsrsv2_analysis(handle, dir, trans, mb, nnzb, descr,
                             bsrVal_device, bsrRowPtr_device, bsrColInd_device, blockDim,
                             info, policy, pBuffer);
// L has unit diagonal, so no structural zero is reported.
    cusparseStatus_t status = cusparseXbsrsv2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }

// step 5: solve L*y = x
    cusparseZbsrsv2_solve(handle, dir, trans, mb, nnzb, &alpha, descr,
                          bsrVal_device, bsrRowPtr_device, bsrColInd_device, blockDim, info,
                          x_device, y_device, policy, pBuffer);
// L has unit diagonal, so no numerical zero is reported.
    status = cusparseXbsrsv2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    CHECK_CUDA(cudaMemcpy(y, y_device, (mb*blockDim) * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

// step 6: free resources
    CHECK_CUDA(cudaFree(bsrVal_device));
    CHECK_CUDA(cudaFree(bsrColInd_device));
    CHECK_CUDA(cudaFree(bsrRowPtr_device));
    CHECK_CUDA(cudaFree(x_device));
    CHECK_CUDA(cudaFree(y_device));
    cudaFree(pBuffer);
    cusparseDestroyBsrsv2Info(info);
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(handle);
}