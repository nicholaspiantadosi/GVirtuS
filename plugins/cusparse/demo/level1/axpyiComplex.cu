#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

void generateSparseVectorFormat(cuComplex *, int, cuComplex *, int *);
void axpyi(cusparseHandle_t, int, int, cuComplex *, cuComplex *, int *, cuComplex *);

int main(void)
{
    srand(time(0));

    int n = 10;
    int nnz = 4;
    cuComplex alpha = make_cuComplex(2, 1);
    cuComplex *xVal = (cuComplex *)malloc(nnz*sizeof(cuComplex));
    int *xInd = (int *)malloc(nnz*sizeof(int));
    cuComplex *y_result = (cuComplex *)malloc(n*sizeof(cuComplex));

    cuComplex *vector = (cuComplex *)malloc(n*sizeof(cuComplex));

    initializeArrayToZerocuComplex(vector, n, nnz);
    initializeArrayToZerocuComplex(y_result, n);

    printf("Array x:\n");
    stampaArrayC(vector, n);

    generateSparseVectorFormat(vector, n, xVal, xInd);

    printf("Array x in formato sparso:\n");
    printf("\txVal: ");
    stampaArrayC(xVal, nnz);
    printf("\n\txInd: ");
    stampaArray(xInd, nnz);
    printf("\n");

    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    axpyi(handle, n, nnz, &alpha, xVal, xInd, y_result);

    printf("Vettore risultato:\n");
    stampaArrayC(y_result, n);

    free(vector);
    free(y_result);
    free(xVal);
    free(xInd);
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return 0;
}

void axpyi(cusparseHandle_t handle, int n, int nnz, cuComplex * alpha, cuComplex * xVal, int * xInd, cuComplex * y) {

    // Variabili su device
    cuComplex * xVal_device;
    int * xInd_device;
    cuComplex * y_device;

    // Allocazione memoria su device
    CHECK_CUDA(cudaMalloc((void**) &xVal_device, nnz * sizeof(cuComplex)));
    CHECK_CUDA(cudaMalloc((void**) &xInd_device, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &y_device, n * sizeof(cuComplex)));

    // Inizializzazione variabili su device
    CHECK_CUDA(cudaMemcpy(xVal_device, xVal, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(xInd_device, xInd, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(y_device, y, n * sizeof(cuComplex), cudaMemcpyHostToDevice));

    // Calcolo prodotto tramite function axpyi
    CHECK_CUSPARSE(cusparseCaxpyi(handle, nnz, alpha, xVal_device, xInd_device, y_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(cuComplex), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));

}

void generateSparseVectorFormat(cuComplex * vector_host, int n, cuComplex * sparse_values, int * sparse_col_index) {
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