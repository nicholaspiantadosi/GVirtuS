#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void generateSparseVectorFormat(float *, int, float *, int *);
void sctr(cusparseHandle_t, int, int, float *, int *, float *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    float *xVal = (float *)malloc(nnz*sizeof(float));
    int *xInd = (int *)malloc(nnz*sizeof(int));
    float *y_result = (float *)malloc(n*sizeof(float));

    float *vector = (float *)malloc(n*sizeof(float));

    initializeArrayRandomSparse(vector, n, nnz);
    initializeArrayToZero(y_result, n);

    printf("Array x:\n");
    stampaArrayF(vector, n);

    generateSparseVectorFormat(vector, n, xVal, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayF(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    sctr(handle, n, nnz, xVal, xInd, y_result);

    printf("Vettore risultato:\n");
    stampaArrayF(y_result, n);

    free(vector);
    free(y_result);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void sctr(cusparseHandle_t handle, int n, int nnz, float * xVal, int * xInd, float * y) {

    // Variabili su device
    float * xVal_device;
    int * xInd_device;
    float * y_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(float)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUSPARSE(cusparseSsctr(handle, nnz, xVal_device, xInd_device, y_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(float), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));

}

void generateSparseVectorFormat(float * vector_host, int n, float * sparse_values, int * sparse_col_index) {
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