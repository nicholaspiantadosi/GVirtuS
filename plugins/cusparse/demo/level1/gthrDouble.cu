#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void gthr(cusparseHandle_t, int, int, double *, double *, int *);
void generateSparseIndex(double *, int, int *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    double *y = (double *)malloc(n*sizeof(double));
    double *xVal = (double *)malloc(nnz*sizeof(double));
    int *xInd = (int *)malloc(nnz*sizeof(int));

    initializeArrayRandomSparseDouble(y, n, nnz);
    initializeArrayToZeroDouble(xVal, nnz);

    printf("Array y:\n");
    stampaArrayD(y, n);

    generateSparseIndex(y, n, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayD(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    gthr(handle, n, nnz, y, xVal, xInd);

    printf("Vettore risultato:\n");
    stampaArrayD(xVal, nnz);

    free(y);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void gthr(cusparseHandle_t handle, int n, int nnz, double * y, double * xVal, int * xInd) {

    // Variabili su device
    double * y_device;
    double * xVal_device;
    int * xInd_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));

    // Scrivi i valori sparsi in vettore denso
    CHECK_CUSPARSE(cusparseDgthr(handle, nnz, y_device, xVal_device, xInd_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(xVal, xVal_device, nnz * sizeof(double), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));
}

void generateSparseIndex(double * vector_host, int n, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i] > 0) {
            sparse_col_index[j] = i;
            ++j;
        }
    }
}