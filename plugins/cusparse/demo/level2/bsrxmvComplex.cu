#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

// Interfacce che usano funcion CUSPARSE
void mat2csr(cusparseHandle_t, cusparseMatDescr_t, cuComplex *, int, int, int, cuComplex *, int *, int *);
void csrmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, cuComplex, cuComplex *, int *, int *, cuComplex *, cuComplex, cuComplex *);
void bsrnnz(cusparseHandle_t, cusparseMatDescr_t, int *, int *, int, int, int, int, int, int, int *, int &);
void csr2bsr(cusparseHandle_t, cusparseMatDescr_t, cuComplex *, int *, int *, int, int, int, int, int , int , int, int *,  cuComplex *, int *);
void bsrxmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, int, int, int, int, cuComplex*, cuComplex*, int*, int*, int*, int*, int, cuComplex*, cuComplex*, cuComplex*);

int main(int argn, char *argv[])
{
    // Variabili generiche
    int m, n, nnz, blockDim, mb=0, nb=0, nnzb=0, sizeOfMask=0;

    // Variabili su host
    cuComplex *matrix_host;
    cuComplex *csr_values_result;
    int *csr_columns_result, *csr_offsets_result;
    cuComplex *bsr_values_result;
    int *bsr_mask, *bsr_columns_result, *bsr_offsets_result, *bsr_offsets_result_start, *bsr_offsets_result_end;
    cuComplex *x_host, *y_result_bsr;
    cuComplex alpha, beta;
    cuComplex *matrix_host_sequential;
    cuComplex *y_result_sequential;

    srand(time(0));

    m = 4;
    n = 5;
    nnz = 9;
    blockDim = 2;
    alpha = make_cuComplex(3, 0);
    beta = make_cuComplex(2, 0);
    sizeOfMask = 1;

    // Allocazione memoria sull'host
    matrix_host = (cuComplex *)malloc((m*n+1)*sizeof(cuComplex));
    csr_values_result=(cuComplex *)malloc(nnz * sizeof(cuComplex));
    csr_offsets_result=(int *)malloc((m + 1) * sizeof(int));
    csr_columns_result=(int *)malloc(nnz * sizeof(int));
    x_host = (cuComplex *)malloc(n*sizeof(cuComplex));
    y_result_bsr = (cuComplex *)malloc(m*sizeof(cuComplex));
    matrix_host_sequential = (cuComplex *)malloc((m*n+1)*sizeof(cuComplex));
    y_result_sequential = (cuComplex *)malloc(m*sizeof(cuComplex));

    // Inizializzazione variabili sull'host
    initializeMatrixRandomSparseC(matrix_host, m, n, nnz);
    initializeArrayRandomC(x_host, n);
    if (beta.x > 0)
        initializeArrayRandomC(y_result_bsr, m);
    else
        initializeArrayToZeroC(y_result_bsr, m);
    copyArrayC(y_result_bsr, y_result_sequential, m);

    // Swap formato matrice per calcolo sequenziale su host
    swapMatrixC(matrix_host, m, n, matrix_host_sequential);

    printf("Matrice sparsa:\n");
    stampaMatrixC(matrix_host, m, n);
    printf("\n");

    printf("Array x:\n");
    stampaArrayC(x_host, n);
    printf("\n");

    printf("Matrice in formato denso per calcolo sequenziale:\n");
    stampaMatrixC1D(matrix_host_sequential, m, n);
    printf("\n");

    printf("Array y:\n");
    stampaArrayC(y_result_bsr, m);
    printf("\n");

    // Dichiarazione dell'handle per CUSPARSE
    cusparseHandle_t handle;

    // Creazione dell'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Creazione della struttura della matrice con relative proprietà e forma
    cusparseMatDescr_t descr = 0;
    cusparseCreateMatDescr(&descr);
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

    // Conversione matrice in formato CSR
    mat2csr(handle, descr, matrix_host, m, n, nnz, csr_values_result, csr_offsets_result, csr_columns_result);

    //Stampa matrice convertita in formato CSR
    printf("Matrice sparsa convertita in formato CSR\n");
    printf("\t csrValA:\t");
    stampaArrayC(csr_values_result, nnz);
    printf("\t csrRowPtrA:\t");
    stampaArray(csr_offsets_result, (m + 1));
    printf("\t csrColIndA:\t");
    stampaArray(csr_columns_result, nnz);
    printf("\n");

    // Calcolo mb e nb a partire da blockDim
    mb = (m + blockDim - 1) / blockDim;
    nb = (n + blockDim - 1) / blockDim;

    // Allocazione variabili su host
    bsr_offsets_result=(int *)malloc((mb + 1) * sizeof(int));
    bsr_offsets_result_start=(int *)malloc((mb) * sizeof(int));
    bsr_offsets_result_end=(int *)malloc((mb) * sizeof(int));

    // Calcolo bsrRowPtrC e blocchi diversi da zero
    bsrnnz(handle, descr, csr_offsets_result, csr_columns_result, m, n, nnz, blockDim, mb, nb, bsr_offsets_result, nnzb);

    // Allocazione variabili su host sulla base del numero di blocchi diversi da zero
    bsr_columns_result=(int *)malloc(nnzb * sizeof(int));
    bsr_mask=(int *)malloc((sizeOfMask) * sizeof(int));
    bsr_values_result=(cuComplex *)malloc((blockDim * blockDim) * nnzb * sizeof(cuComplex));

    // Conversione da formato CSR a BSR
    csr2bsr(handle, descr, csr_values_result, csr_offsets_result, csr_columns_result, m, n, nnz, blockDim, mb, nb, nnzb, bsr_offsets_result, bsr_values_result, bsr_columns_result);

    initializeArrayTo2(bsr_mask, sizeOfMask);
    for(int i=0;i<mb;i++)
    {
        bsr_offsets_result_start[i]=bsr_offsets_result[i];
        bsr_offsets_result_end[mb - i - 1]=bsr_offsets_result[mb - i];
    }

    // Stampa matrice convertita in formato BSRX
    printf("Matrice sparsa convertita in formato BSRX\n");
    printf("\t bsrVal:\t");
    stampaArrayC(bsr_values_result, (blockDim * blockDim) * nnzb);
    printf("\t bsrMaskPtr:\t");
    stampaArray(bsr_mask, sizeOfMask);
    printf("\t bsrRowPtr:\t");
    stampaArray(bsr_offsets_result_start, (mb));
    printf("\t bsrRowPtrEnd:\t");
    stampaArray(bsr_offsets_result_end, (mb));
    printf("\t bsrColInd:\t");
    stampaArray(bsr_columns_result, nnzb);
    printf("\n");

    bsrxmv(handle, descr, m, n, nnz, sizeOfMask, mb, nb, nnzb, &alpha, bsr_values_result, bsr_mask, bsr_offsets_result_start, bsr_offsets_result_end, bsr_columns_result, blockDim, x_host, &beta, y_result_bsr);

    //Stampa array risultato dall'operazione bsrxmv tra matrice in formato BSR, vettore x_host, alpha e beta
    printf("Vettore risultato dall'operazione bsrxmv\n");
    stampaArrayC(y_result_bsr, m);
    printf("\n");

    //Libera la memoria sull'host
    free(csr_values_result);
    free(csr_offsets_result);
    free(csr_columns_result);
    free(bsr_values_result);
    free(bsr_mask);
    free(bsr_offsets_result);
    free(bsr_offsets_result_start);
    free(bsr_offsets_result_end);
    free(bsr_columns_result);
    free(matrix_host);
    free(matrix_host_sequential);

    // Termina l'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

}

void mat2csr(cusparseHandle_t handle, cusparseMatDescr_t descr, cuComplex * matrix, int m, int n, int nnz, cuComplex * csrValA, int * csrRowPtrA, int * csrColIndA)
{
    int nnz_total = 0;

    // Variabili su device
    cuComplex *matrix_device;
    cuComplex *csr_values_device;
    int *csr_columns_device, *csr_offsets_device;
    int *nnz_per_row;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &matrix_device, m * n * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csr_values_device, nnz * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csr_offsets_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csr_columns_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &nnz_per_row, m * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(matrix_device, matrix, m * n * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Calcolo valori diversi da zero
    CHECK_CUSPARSE(cusparseCnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descr, matrix_device, m, nnz_per_row, &nnz_total));

    // Controllo su valori diversi da zero richiesti in input rispetto a quelli calcolati tramite cusparseCnnz()
    if (nnz != nnz_total) {
        printf("I valori diversi da zero richiesti in input sono diversi rispetto a quelli rilevati: richiesti %d valori ma sono stati rilevati %d valori diversi da zero!\n\n", nnz, nnz_total);
        exit(EXIT_FAILURE);
    }

    // Conversione matrice in formato CSR
    CHECK_CUSPARSE(cusparseCdense2csr(handle, m, n, descr, matrix_device, m, nnz_per_row, csr_values_device, csr_offsets_device, csr_columns_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(csrValA, csr_values_device, nnz * sizeof(cuComplex), cudaMemcpyDeviceToHost));
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

void csr2bsr(cusparseHandle_t handle, cusparseMatDescr_t descr, cuComplex * csrValA, int * csrRowPtrA, int * csrColIndA, int m, int n, int nnz, int blockDim, int mb, int nb, int nnzb, int * bsrRowPtrC, cuComplex * bsrValC, int * bsrColIndC)
{
    // Variabili su device
    cuComplex *bsrValC_device;
    int *bsrRowPtrC_device, *bsrColIndC_device;
    cuComplex *csrValA_device;
    int *csrRowPtrA_device, *csrColIndA_device;

    // Allocazione variabili su device sulla base del numero di blocchi diversi da zero
    CHECK_CUDA(cudaMalloc((void**) &csrValA_device, nnz * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &csrRowPtrA_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csrColIndA_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrColIndC_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**)&bsrRowPtrC_device, (mb + 1) * sizeof(int)));

    // Copia da host a device
    CHECK_CUDA(cudaMemcpy(csrValA_device, csrValA, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrRowPtrA_device, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrColIndA_device, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtrC_device, bsrRowPtrC, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Conversione da CSR a BSR
    CHECK_CUSPARSE(cusparseCcsr2bsr(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrValA_device, csrRowPtrA_device, csrColIndA_device, blockDim, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(bsrValC, bsrValC_device, ((blockDim * blockDim) * nnzb) * sizeof(cuComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bsrColIndC, bsrColIndC_device, nnzb * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(bsrValC_device));
    CHECK_CUDA(cudaFree(bsrColIndC_device));
    //CHECK_CUDA(cudaFree(bsrRowPtrC_device));
    CHECK_CUDA(cudaFree(csrValA_device));
    CHECK_CUDA(cudaFree(csrRowPtrA_device));
    CHECK_CUDA(cudaFree(csrColIndA_device));

}

void bsrxmv(cusparseHandle_t handle, cusparseMatDescr_t descr, int m, int n, int nnz, int sizeOfMask, int mb, int nb, int nnzb, cuComplex* alpha, cuComplex* bsrVal, int* bsrMaskPtr, int* bsrRowPtr, int* bsrEndPtr, int* bsrColInd, int blockDim, cuComplex* x, cuComplex* beta, cuComplex* y)
{
    // Variabili su device
    cuComplex *x_device, *y_device;
    cuComplex *bsrVal_device;
    int *bsrMaskPtr_device, *bsrRowPtr_device, *bsrEndPtr_device, *bsrColInd_device;

    // Allocazione variabili su device
    CHECK_CUDA(cudaMalloc((void**) &bsrVal_device, (blockDim * blockDim) * nnzb * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &bsrMaskPtr_device, sizeOfMask * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &bsrRowPtr_device, mb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &bsrEndPtr_device, mb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &bsrColInd_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&x_device, (nb*blockDim) * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**)&y_device, (mb*blockDim) * sizeof(cuComplex)));

    // Copia valori da host a device
    CHECK_CUDA(cudaMemcpy(x_device, x, n * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, m * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrVal_device, bsrVal, (blockDim * blockDim) * nnzb * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrMaskPtr_device, bsrMaskPtr, sizeOfMask * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtr_device, bsrRowPtr, mb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrEndPtr_device, bsrEndPtr, mb * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrColInd_device, bsrColInd, nnzb * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseCbsrxmv(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, sizeOfMask, mb, nb, nnzb, alpha, descr, bsrVal_device, bsrMaskPtr_device, bsrRowPtr_device, bsrEndPtr_device, bsrColInd_device, blockDim, x_device, beta, y_device));

    CHECK_CUDA(cudaMemcpy(y, y_device, (mb*blockDim) * sizeof(cuComplex), cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaFree(bsrVal_device));
    CHECK_CUDA(cudaFree(bsrMaskPtr_device));
    CHECK_CUDA(cudaFree(bsrColInd_device));
    CHECK_CUDA(cudaFree(bsrRowPtr_device));
    CHECK_CUDA(cudaFree(bsrEndPtr_device));
    CHECK_CUDA(cudaFree(x_device));
    CHECK_CUDA(cudaFree(y_device));

}