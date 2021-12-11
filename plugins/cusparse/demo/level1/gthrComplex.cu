#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void gthr(cusparseHandle_t, int, int, cuComplex *, cuComplex *, int *);
void generateSparseIndex(cuComplex *, int, int *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    cuComplex *y = (cuComplex *)malloc(n*sizeof(cuComplex));
    cuComplex *xVal = (cuComplex *)malloc(nnz*sizeof(cuComplex));
    int *xInd = (int *)malloc(nnz*sizeof(int));

    initializeArrayRandomSparsecuComplex(y, n, nnz);
    initializeArrayToZerocuComplex(xVal, nnz);

    printf("Array y:\n");
    stampaArrayC(y, n);

    generateSparseIndex(y, n, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayC(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    gthr(handle, n, nnz, y, xVal, xInd);

    printf("Vettore risultato:\n");
    stampaArrayC(xVal, nnz);

    free(y);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void gthr(cusparseHandle_t handle, int n, int nnz, cuComplex * y, cuComplex * xVal, int * xInd) {

    // Variabili su device
    cuComplex * y_device;
    cuComplex * xVal_device;
    int * xInd_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Scrivi i valori sparsi in vettore denso
    CHECK_CUSPARSE(cusparseCgthr(handle, nnz, y_device, xVal_device, xInd_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(xVal, xVal_device, nnz * sizeof(cuComplex), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));
}

void generateSparseIndex(cuComplex * vector_host, int n, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i].x > 0) {
            sparse_col_index[j] = i;
            ++j;
        }
    }
}