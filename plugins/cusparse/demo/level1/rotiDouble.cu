#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void generateSparseVectorFormat(double *, int, double *, int *);
void roti(cusparseHandle_t , int , int , double * , int * , double * , double * , double * );

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    double c = 2;
    double s = 1;
    double *xVal = (double *)malloc(nnz*sizeof(double));
    int *xInd = (int *)malloc(nnz*sizeof(int));
    double *y_result = (double *)malloc(n*sizeof(double));

    double *vector = (double *)malloc(n*sizeof(double));

    initializeArrayRandomSparseDouble(vector, n, nnz);
    initializeArrayToZeroDouble(y_result, n);

    printf("Array x:\n");
    stampaArrayD(vector, n);

    generateSparseVectorFormat(vector, n, xVal, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayD(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    roti(handle, n, nnz, xVal, xInd, y_result, &c, &s);

    printf("Vettore risultato:\n");
    stampaArrayD(y_result, n);

    printf("Vettore x:\n");
    stampaArrayD(xVal, nnz);

    free(vector);
    free(y_result);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void roti(cusparseHandle_t handle, int n, int nnz, double * xVal, int * xInd, double * y, double * c, double * s) {

    // Variabili su device
    double * xVal_device;
    int * xInd_device;
    double * y_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(double)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(double), cudaMemcpyHostToDevice));

    // Calcolo prodotto tramite function axpyi
    CHECK_CUSPARSE(cusparseDroti(handle, nnz, xVal_device, xInd_device, y_device, c, s, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(xVal, xVal_device, nnz * sizeof(double), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));

}

void generateSparseVectorFormat(double * vector_host, int n, double * sparse_values, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i] > 0) {
            sparse_values[j] = vector_host[i];
            sparse_col_index[j] = i;
            ++j;
        }
    }
}