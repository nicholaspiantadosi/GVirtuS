#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

// Interfacce che usano funcion CUSPARSE
void mat2csr(cusparseHandle_t, cusparseMatDescr_t, float *, int, int, int, float *, int *, int *);
void csrmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, float, float *, int *, int *, float *, float, float *);
void bsrnnz(cusparseHandle_t, cusparseMatDescr_t, int *, int *, int, int, int, int, int, int, int *, int &);
void csr2bsr(cusparseHandle_t, cusparseMatDescr_t, float *, int *, int *, int, int, int, int, int , int , int, int *,  float *, int *);
void bsrmv(cusparseHandle_t, cusparseMatDescr_t, int, int, int, int, int, int, float, float *, int *, int *, int, float *, float, float *);

int main(int argn, char *argv[])
{
    // Variabili generiche
    int m, n, nnz, blockDim, mb=0, nb=0, nnzb=0;

    // Variabili su host
    float *matrix_host;
    float *csr_values_result;
    int *csr_columns_result, *csr_offsets_result;
    float *bsr_values_result;
    int *bsr_columns_result, *bsr_offsets_result;
    float *x_host, *y_result_bsr;
    float alpha, beta;
    float *matrix_host_sequential;
    float *y_result_sequential;

    srand(time(0));

    m = 4;
    n = 5;
    nnz = 9;
    blockDim = 2;
    alpha = 3;
    beta = 2;

    // Allocazione memoria sull'host
    matrix_host = (float *)malloc((m*n+1)*sizeof(float));
    csr_values_result=(float *)malloc(nnz * sizeof(float));
    csr_offsets_result=(int *)malloc((m + 1) * sizeof(int));
    csr_columns_result=(int *)malloc(nnz * sizeof(int));
    x_host = (float *)malloc(n*sizeof(float));
    y_result_bsr = (float *)malloc(m*sizeof(float));
    matrix_host_sequential = (float *)malloc((m*n+1)*sizeof(float));
    y_result_sequential = (float *)malloc(m*sizeof(float));

    // Inizializzazione variabili sull'host
    initializeMatrixRandomSparseF(matrix_host, m, n, nnz);
    initializeArrayRandomF(x_host, n);
    if (beta > 0)
        initializeArrayRandomF(y_result_bsr, m);
    else
        initializeArrayToZeroF(y_result_bsr, m);
    copyArrayF(y_result_bsr, y_result_sequential, m);

    // Swap formato matrice per calcolo sequenziale su host
    swapMatrixF(matrix_host, m, n, matrix_host_sequential);

    printf("Matrice sparsa:\n");
    stampaMatrixF(matrix_host, m, n);
    printf("\n");

    printf("Array x:\n");
    stampaArrayF(x_host, n);
    printf("\n");

    printf("Matrice in formato denso per calcolo sequenziale:\n");
    stampaMatrixF1D(matrix_host_sequential, m, n);
    printf("\n");

    printf("Array y:\n");
    stampaArrayF(y_result_bsr, m);
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
    stampaArrayF(csr_values_result, nnz);
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

    // Calcolo bsrRowPtrC e blocchi diversi da zero
    bsrnnz(handle, descr, csr_offsets_result, csr_columns_result, m, n, nnz, blockDim, mb, nb, bsr_offsets_result, nnzb);

    // Allocazione variabili su host sulla base del numero di blocchi diversi da zero
    bsr_columns_result=(int *)malloc(nnzb * sizeof(int));
    bsr_values_result=(float *)malloc((blockDim * blockDim) * nnzb * sizeof(float));

    // Conversione da formato CSR a BSR
    csr2bsr(handle, descr, csr_values_result, csr_offsets_result, csr_columns_result, m, n, nnz, blockDim, mb, nb, nnzb, bsr_offsets_result, bsr_values_result, bsr_columns_result);

    // Stampa matrice convertita in formato BSR
    printf("Matrice sparsa convertita in formato BSR\n");
    printf("\t bsrValC:\t");
    stampaArrayF(bsr_values_result, (blockDim * blockDim) * nnzb);
    printf("\t bsrRowPtrC:\t");
    stampaArray(bsr_offsets_result, (mb + 1));
    printf("\t bsrColIndC:\t");
    stampaArray(bsr_columns_result, nnzb);
    printf("\n");

    // Operazione bsrmv corrispondente al seguente prodotto y = alpha * A * x + beta * y
    bsrmv(handle, descr, m, n, nnz, mb, nb, nnzb, alpha, bsr_values_result, bsr_offsets_result, bsr_columns_result, blockDim, x_host, beta, y_result_bsr);

    //Stampa array risultato dall'operazione bsrmv tra matrice in formato BSR, vettore x_host, alpha e beta
    printf("Vettore risultato dall'operazione bsrmv\n");
    stampaArrayF(y_result_bsr, m);
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

// Function che consente di ottenere la matrice in formato CSR
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - matrix - matrice sparsa
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//     output:
//         - csrValA - array contenente i valori
//         - csrRowPtrA - array contenente gli indici per i quali bisogna considerare il nuovo indice di riga
//         - csrColIndA - array contenente gli indici di colonne
void mat2csr(cusparseHandle_t handle, cusparseMatDescr_t descr, float * matrix, int m, int n, int nnz, float * csrValA, int * csrRowPtrA, int * csrColIndA)
{
    int nnz_total = 0;

    // Variabili su device
    float *matrix_device;
    float *csr_values_device;
    int *csr_columns_device, *csr_offsets_device;
    int *nnz_per_row;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &matrix_device, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &csr_values_device, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &csr_offsets_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csr_columns_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &nnz_per_row, m * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(matrix_device, matrix, m * n * sizeof(float), cudaMemcpyHostToDevice));

    // Calcolo valori diversi da zero
    CHECK_CUSPARSE(cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descr, matrix_device, m, nnz_per_row, &nnz_total));

    // Controllo su valori diversi da zero richiesti in input rispetto a quelli calcolati tramite cusparseSnnz()
    if (nnz != nnz_total) {
        printf("I valori diversi da zero richiesti in input sono diversi rispetto a quelli rilevati: richiesti %d valori ma sono stati rilevati %d valori diversi da zero!\n\n", nnz, nnz_total);
        exit(EXIT_FAILURE);
    }

    // Conversione matrice in formato CSR
    CHECK_CUSPARSE(cusparseSdense2csr(handle, m, n, descr, matrix_device, m, nnz_per_row, csr_values_device, csr_offsets_device, csr_columns_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(csrValA, csr_values_device, nnz * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csrRowPtrA, csr_offsets_device, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(csrColIndA, csr_columns_device, nnz * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(csr_values_device));
    CHECK_CUDA(cudaFree(csr_offsets_device));
    CHECK_CUDA(cudaFree(csr_columns_device));
    CHECK_CUDA(cudaFree(matrix_device));
    CHECK_CUDA(cudaFree(nnz_per_row));
}

// Function che consente di ottenere il numero dei blocchi con elementi diversi da zero e l'array bsrRowPtrC inizializzato con i valori di indice per i blocchi di riga
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - csrRowPtrA - array contenente gli indici per i quali bisogna considerare il nuovo indice di riga, della matrice in formato CSR
//         - csrColIndA - array contenente gli indici di colonne, della matrice in formato CSR
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//         - blockDim - dimensione che deve avere il blocco nel formato BSR
//         - mb - numero di righe dei blocchi calcolati come segue: mb = (m + blockDim - 1) / blockDim
//         - nb - numero di colonne dei blocchi calcolati come segue: nb = (n + blockDim - 1) / blockDim;
//     output:
//         - bsrRowPtrC - array contenente i valori di indice per i blocchi di riga della matrice in formato BSR
//         - nnzb - blocchi con valori diversi da zero
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

// Function che consente di ottenere il numero dei blocchi con elementi diversi da zero e l'array bsrRowPtrC inizializzato con i valori di indice per i blocchi di riga
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - csrValA - array contenente i valori della matrice in formato CSR
//         - csrRowPtrA - array contenente gli indici per i quali bisogna considerare il nuovo indice di riga, della matrice in formato CSR
//         - csrColIndA - array contenente gli indici di colonne, della matrice in formato CSR
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//         - blockDim - dimensione che deve avere il blocco nel formato BSR
//         - mb - numero di righe dei blocchi calcolati come segue: mb = (m + blockDim - 1) / blockDim
//         - nb - numero di colonne dei blocchi calcolati come segue: nb = (n + blockDim - 1) / blockDim;
//         - nnzb - blocchi con valori diversi da zero
//         - bsrRowPtrC - array contenente i valori di indice per i blocchi di riga della matrice in formato BSR
//     output:
//         - bsrValC - array contenente i valori della matrice in formato BSR
//         - bsrColIndC - array contenente i valori di colonna dei blocchi della matrice in formato BSR
void csr2bsr(cusparseHandle_t handle, cusparseMatDescr_t descr, float * csrValA, int * csrRowPtrA, int * csrColIndA, int m, int n, int nnz, int blockDim, int mb, int nb, int nnzb, int * bsrRowPtrC, float * bsrValC, int * bsrColIndC)
{
    // Variabili su device
    float *bsrValC_device;
    int *bsrRowPtrC_device, *bsrColIndC_device;
    float *csrValA_device;
    int *csrRowPtrA_device, *csrColIndA_device;

    // Allocazione variabili su device sulla base del numero di blocchi diversi da zero
    CHECK_CUDA(cudaMalloc((void**) &csrValA_device, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &csrRowPtrA_device, (m + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &csrColIndA_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrColIndC_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&bsrRowPtrC_device, (mb + 1) * sizeof(int)));

    // Copia da host a device
    CHECK_CUDA(cudaMemcpy(csrValA_device, csrValA, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrRowPtrA_device, csrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(csrColIndA_device, csrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtrC_device, bsrRowPtrC, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Conversione da CSR a BSR
    CHECK_CUSPARSE(cusparseScsr2bsr(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrValA_device, csrRowPtrA_device, csrColIndA_device, blockDim, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(bsrValC, bsrValC_device, ((blockDim * blockDim) * nnzb) * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(bsrColIndC, bsrColIndC_device, nnzb * sizeof(int), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(bsrValC_device));
    CHECK_CUDA(cudaFree(bsrColIndC_device));
    //CHECK_CUDA(cudaFree(bsrRowPtrC_device));
    CHECK_CUDA(cudaFree(csrValA_device));
    CHECK_CUDA(cudaFree(csrRowPtrA_device));
    CHECK_CUDA(cudaFree(csrColIndA_device));

}

// Function che calcola il prodotto matrice per vettore tramite bsrmv (y = alpha * A * x + beta * y)
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//         - mb - numero di righe dei blocchi calcolati come segue: mb = (m + blockDim - 1) / blockDim
//         - nb - numero di colonne dei blocchi calcolati come segue: nb = (n + blockDim - 1) / blockDim;
//         - nnzb - blocchi con valori diversi da zero
//         - alpha - scalare che viene utilizzato nel primo prodotto bsrmv (alpha * A * x)
//         - bsrValC - array contenente i valori della matrice in formato BSR
//         - bsrRowPtrC - array contenente i valori di indice per i blocchi di riga della matrice in formato BSR
//         - bsrColIndC - array contenente i valori di colonna dei blocchi della matrice in formato BSR
//         - blockDim - dimensione che deve avere il blocco nel formato BSR
//         - x - array che viene utilizzato nel primo prodotto bsrmv (alpha * A * x)
//         - beta - scalare che viene utilizzato nel secondo prodotto bsrmv (beta * y)
//         - y - array che viene utilizzato nel secondo prodotto bsrmv (beta * y)
//     output:
//         - y - array risultato dell'operazione bsrmv (y = alpha * A * x + beta * y)
void bsrmv(cusparseHandle_t handle, cusparseMatDescr_t descr, int m, int n, int nnz, int mb, int nb, int nnzb, float alpha, float * bsrValC, int * bsrRowPtrC, int * bsrColIndC, int blockDim, float * x, float beta, float * y)
{
    // Variabili su device
    float *x_device, *y_device;
    float *bsrValC_device;
    int *bsrRowPtrC_device, *bsrColIndC_device;

    // Allocazione variabili su device
    CHECK_CUDA(cudaMalloc((void**) &bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &bsrRowPtrC_device, (mb + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &bsrColIndC_device, nnzb * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&x_device, (nb*blockDim) * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&y_device, (mb*blockDim) * sizeof(float)));

    // Copia valori da host a device
    CHECK_CUDA(cudaMemcpy(x_device, x, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, m * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrValC_device, bsrValC, (blockDim * blockDim) * nnzb * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrRowPtrC_device, bsrRowPtrC, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(bsrColIndC_device, bsrColIndC, nnzb * sizeof(int), cudaMemcpyHostToDevice));

    // Operazione y = alpha * A * x + beta * y
    // la matrice A è rappresentata in formato BSR dagli array bsrValC_device, bsrRowPtrC_device e bsrColIndC_device
    CHECK_CUSPARSE(cusparseSbsrmv(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device, blockDim, x_device, &beta, y_device));

    // Copia valori da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, (mb*blockDim) * sizeof(float), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(bsrValC_device));
    CHECK_CUDA(cudaFree(bsrColIndC_device));
    CHECK_CUDA(cudaFree(bsrRowPtrC_device));
    CHECK_CUDA(cudaFree(x_device));
    CHECK_CUDA(cudaFree(y_device));

}