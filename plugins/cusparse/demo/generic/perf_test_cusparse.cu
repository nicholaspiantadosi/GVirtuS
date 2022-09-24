#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utilities2.h"

#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

// Interfacce che usano funcion CUSPARSE
void generateSparseVectorFormat(double *, int, double *, int *);
void axpyi(cusparseHandle_t, int, int, double *, double *, int *, double *);

void stampaArray(double*, int);
void initializeArray(double *, int);
void initializeArray(double *, int, int);

void execute(int, int, double, int);

int main(int argn, char *argv[])
{

    std::cout << "n;allocation (micros);execution (micros);free (micros);total (micros);" << std::endl;
/*
    execute(10, 1, 2, 1);
    execute(100, 10, 2, 1);
    execute(1000, 100, 2, 1);
    execute(10000, 1000, 2, 1);
    execute(100000, 10000, 2, 1);
    execute(1000000, 100000, 2, 1);
    execute(10000000, 1000000, 2, 1);
    execute(100000000, 10000000, 2, 1);
    execute(1000000000, 100000000, 2, 1);
*/
    execute(10, 0, 2, 1);
    execute(100, 1, 2, 1);
    execute(1000, 10, 2, 1);
    execute(10000, 100, 2, 1);
    execute(100000, 1000, 2, 1);
    execute(1000000, 10000, 2, 1);
    execute(10000000, 100000, 2, 1);
    //execute(100000000, 1000000, 2, 1);
    //execute(1000000000, 10000000, 2, 1);
}

void execute(int n, int nnz, double alpha, int yRandom) {
    // Inizio esecuzione
    auto t1 = Clock::now();

    // Variabili su host
    double *vector_host;
    double *y_result;
    double *sparse_values;
    int *sparse_col_index;

    // Imposto il seed in base all'orario
    srand(time(0));

    // Allocazione memoria sull'host
    vector_host = (double *)malloc(n*sizeof(double));
    y_result = (double *)malloc(n*sizeof(double));
    sparse_values = (double *)malloc(nnz*sizeof(double));
    sparse_col_index = (int *)malloc(nnz*sizeof(int));

    // Inizializzazione variabili sull'host
    /*
    initializeArray(vector_host, n);
    initializeArrayToZero(y_result, n);
    if(yRandom > 0) {
        initializeArray(y_result, n, yRandom);
    }
    */
    initializeArrayRandomSparse(vector_host, n, nnz);
    initializeArrayToZero(y_result, n);
    if(yRandom > 0) {
        initializeArrayRandom(y_result, n);
    }

    // Calcolo array in formato sparso
    generateSparseVectorFormat(vector_host, n, sparse_values, sparse_col_index);

    // Dichiarazione dell'handle per CUSPARSE
    cusparseHandle_t handle;

    // Creazione dell'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseCreate(&handle));

    auto t2 = Clock::now();

    // Prodotto vettore * alpha, il risultato sommato in y (function CUSPARSE axpyi)
    axpyi(handle, n, nnz, &alpha, sparse_values, sparse_col_index, y_result);

    //stampaArray(y_result, 10);

    auto t3 = Clock::now();

    //Libera la memoria sull'host
    free(vector_host);
    free(y_result);
    free(sparse_values);
    free(sparse_col_index);

    // Termina l'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseDestroy(handle));

    auto t4 = Clock::now();

    // Fine esecuzione
    /*
    std::cout << "axpyi - y = y + alpha * x" << std::endl;
    //std::cout << "y = y + " << alpha << " * x[" << n << "] {nnz = " << nnz << "}" << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << "Time to allocate resources and prepare data: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << std::endl;
    std::cout << "Time to execute axpyi using GPU:             " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " microseconds" << std::endl;
    std::cout << "Time to free resources:                      " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << std::endl;
    //std::cout << "Time to check results:                       " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << " microseconds" << std::endl;
    std::cout << "Time total:                                  " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << " microseconds" << std::endl;
    */
    std::cout << n << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << ";" << std::endl;
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

void axpyi(cusparseHandle_t handle, int n, int nnz, double * alpha, double * xVal, int * xInd, double * y) {

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
    CHECK_CUSPARSE(cusparseDaxpyi(handle, nnz, alpha, xVal_device, xInd_device, y_device, CUSPARSE_INDEX_BASE_ZERO));

    // Copia risultato da device a host
    CHECK_CUDA(cudaMemcpy(y, y_device, n * sizeof(double), cudaMemcpyDeviceToHost));

    //Libera la memoria sul device
    CHECK_CUDA(cudaFree(xVal_device));
    CHECK_CUDA(cudaFree(xInd_device));
    CHECK_CUDA(cudaFree(y_device));

}

// Function base
void initializeArray(double *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = i;
}

void initializeArray(double *array, int n, int yRandom)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = i+yRandom;
}

void stampaArray(double* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
}
