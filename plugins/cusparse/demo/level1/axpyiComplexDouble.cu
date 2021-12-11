#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void generateSparseVectorFormat(cuDoubleComplex *, int, cuDoubleComplex *, int *);
void axpyi(cusparseHandle_t, int, int, cuDoubleComplex *, cuDoubleComplex *, int *, cuDoubleComplex *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    cuDoubleComplex alpha = make_cuDoubleComplex(2, 1);
    cuDoubleComplex *xVal = (cuDoubleComplex *)malloc(nnz*sizeof(cuDoubleComplex));
    int *xInd = (int *)malloc(nnz*sizeof(int));
    cuDoubleComplex *y_result = (cuDoubleComplex *)malloc(n*sizeof(cuDoubleComplex));

    cuDoubleComplex *vector = (cuDoubleComplex *)malloc(n*sizeof(cuDoubleComplex));

    initializeArrayToZerocuDoubleComplex(vector, n, nnz);
    initializeArrayToZerocuDoubleComplex(y_result, n);

    printf("Array x:\n");
    stampaArrayZ(vector, n);

    generateSparseVectorFormat(vector, n, xVal, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayZ(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    axpyi(handle, n, nnz, &alpha, xVal, xInd, y_result);

    printf("Vettore risultato:\n");
    stampaArrayZ(y_result, n);

    free(vector);
    free(y_result);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void axpyi(cusparseHandle_t handle, int n, int nnz, cuDoubleComplex * alpha, cuDoubleComplex * xVal, int * xInd, cuDoubleComplex * y) {

    // Variabili su device
    cuDoubleComplex * xVal_device;
    int * xInd_device;
    cuDoubleComplex * y_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(cuDoubleComplex)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

    // Calcolo prodotto tramite function axpyi
    CHECK_CUSPARSE(cusparseZaxpyi(handle, nnz, alpha, xVal_device, xInd_device, y_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));

}

void generateSparseVectorFormat(cuDoubleComplex * vector_host, int n, cuDoubleComplex * sparse_values, int * sparse_col_index) {
    int i = 0;
    int j = 0;
    for (i = 0; i < n; i++) {
        if (vector_host[i].x > 0) {
            sparse_values[j] = vector_host[i];
            sparse_col_index[j] = i;
            ++j;
        }
    }
}