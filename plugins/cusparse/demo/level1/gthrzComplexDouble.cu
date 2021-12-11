#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void gthrz(cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *, int *);
void generateSparseIndex(cuDoubleComplex *, int, int *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    cuDoubleComplex *y = (cuDoubleComplex *)malloc(n*sizeof(cuDoubleComplex));
    cuDoubleComplex *xVal = (cuDoubleComplex *)malloc(nnz*sizeof(cuDoubleComplex));
    int *xInd = (int *)malloc(nnz*sizeof(int));

    initializeArrayRandomSparsecuDoubleComplex(y, n, nnz);
    initializeArrayToZerocuDoubleComplex(xVal, nnz);

    printf("Array y:\n");
    stampaArrayZ(y, n);

    generateSparseIndex(y, n, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayZ(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    gthrz(handle, n, nnz, y, xVal, xInd);

    printf("Vettore risultato:\n");
    stampaArrayZ(xVal, nnz);

    printf("Array y:\n");
    stampaArrayZ(y, n);

    free(y);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void gthrz(cusparseHandle_t handle, int n, int nnz, cuDoubleComplex * y, cuDoubleComplex * xVal, int * xInd) {

    // Variabili su device
    cuDoubleComplex * y_device;
    cuDoubleComplex * xVal_device;
    int * xInd_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Scrivi i valori sparsi in vettore denso
    CHECK_CUSPARSE(cusparseZgthrz(handle, nnz, y_device, xVal_device, xInd_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(xVal, xVal_device, nnz * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));
}

void generateSparseIndex(cuDoubleComplex * vector_host, int n, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i].x > 0) {
            sparse_col_index[j] = i;
            ++j;
        }
    }
}