#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void gthr(cusparseHandle_t, int, int, float *, float *, int *);
void generateSparseIndex(float *, int, int *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    float *y = (float *)malloc(n*sizeof(float));
    float *xVal = (float *)malloc(nnz*sizeof(float));
    int *xInd = (int *)malloc(nnz*sizeof(int));

    initializeArrayRandomSparse(y, n, nnz);
    initializeArrayToZero(xVal, nnz);

    printf("Array y:\n");
    stampaArrayF(y, n);

    generateSparseIndex(y, n, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayF(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    gthr(handle, n, nnz, y, xVal, xInd);

    printf("Vettore risultato:\n");
    stampaArrayF(xVal, nnz);

    free(y);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void gthr(cusparseHandle_t handle, int n, int nnz, float * y, float * xVal, int * xInd) {

    // Variabili su device
    float * y_device;
    float * xVal_device;
    int * xInd_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Scrivi i valori sparsi in vettore denso
    CHECK_CUSPARSE(cusparseSgthr(handle, nnz, y_device, xVal_device, xInd_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(xVal, xVal_device, nnz * sizeof(float), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));
}

void generateSparseIndex(float * vector_host, int n, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i] > 0) {
            sparse_col_index[j] = i;
            ++j;
        }
    }
}