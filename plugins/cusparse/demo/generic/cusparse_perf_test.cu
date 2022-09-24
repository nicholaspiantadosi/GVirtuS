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

// Interfacce per calcoli su host
void sommaArrayCompPerCompCPU(double *, double *, int, double *);
void prodottoScalareCPU(double *, double, int, double *);
void axpyiSequential(double *, int, double, double *);

int main(int argn, char *argv[])
{
	// Inizio esecuzione
	clock_t tStart = clock();
    auto t1 = Clock::now();

	// Variabili generiche
	int n, nnz;
	int yRandom;

    // Variabili su host
    double *vector_host;
	double *y_result;
	double *sparse_values;
	int *sparse_col_index;
	double alpha;
	double *y_result_host;

	// Imposto il seed in base all'orario
	srand(time(0));

    n = 1000000;
    nnz = 100;
    alpha = 1;
    yRandom = 0;

    // Allocazione memoria sull'host
	vector_host = (double *)malloc(n*sizeof(double));
	y_result = (double *)malloc(n*sizeof(double));
	sparse_values = (double *)malloc(nnz*sizeof(double));
	sparse_col_index = (int *)malloc(nnz*sizeof(int));
	y_result_host = (double *)malloc(n*sizeof(double));

	// Inizializzazione variabili sull'host
	initializeArrayRandomSparse(vector_host, n, nnz);
	initializeArrayToZero(y_result, n);
	if(yRandom > 0) {
		initializeArrayRandom(y_result, n);
	}
	copyArray(y_result, y_result_host, n);

	// Calcolo array in formato sparso
	generateSparseVectorFormat(vector_host, n, sparse_values, sparse_col_index);

	// Dichiarazione dell'handle per CUSPARSE
	cusparseHandle_t handle;

	// Creazione dell'handle per CUSPARSE
	CHECK_CUSPARSE(cusparseCreate(&handle));

    clock_t tAllocationAndDataPreparation = clock();
    auto t2 = Clock::now();

	// Prodotto vettore * alpha, il risultato sommato in y (function CUSPARSE axpyi)
	axpyi(handle, n, nnz, &alpha, sparse_values, sparse_col_index, y_result);

    clock_t tAxpyi = clock();
    auto t3 = Clock::now();

    // Operazione axpyi in sequenziale
    axpyiSequential(vector_host, n, alpha, y_result_host);

    clock_t tAxpyiSequential = clock();
    auto t4 = Clock::now();

	//Libera la memoria sull'host
	free(vector_host);
	free(y_result);
	free(sparse_values);
	free(sparse_col_index);
	free(y_result_host);

	// Termina l'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseDestroy(handle));

    clock_t tFreeMemory = clock();
    auto t5 = Clock::now();

    // Verifica se i due array risultato (uno calcolato con cusparse sul device e uno calcolato su host) coincidono
    //equalArrayD(y_result, y_result_host, n);

    clock_t tCheck = clock();
    auto t6 = Clock::now();

	// Fine esecuzione
    std::cout << "axpyi - y = y + alpha * x" << std::endl;
    //std::cout << "y = y + " << alpha << " * x[" << n << "] {nnz = " << nnz << "}" << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << "Time to allocate resources and prepare data: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << std::endl;
    std::cout << "Time to execute axpyi using GPU:             " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " microseconds" << std::endl;
    std::cout << "Time to execute axpyi using CPU:             " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << std::endl;
    std::cout << "Time to free resources:                      " << std::chrono::duration_cast<std::chrono::microseconds>(t5 - t4).count() << " microseconds" << std::endl;
    //std::cout << "Time to check results:                       " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << " microseconds" << std::endl;
    std::cout << "Time total:                                  " << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t1).count() << " microseconds" << std::endl;

    /*
    printf("\nTime to allocate resources and prepare data:\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    printf("\nTime to execute axpyi using GPU:\t\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
    printf("\nTime to free resources:\t\t\t\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count());
    printf("\nTime to execute axpyi using CPU:\t\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t5 - t4).count());
    printf("\nTime to check results:\t\t\t\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count());
    printf("\nTime total:\t\t\t\t\t\t\t%.4fns", std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t1).count());
    printf("\n");

    printf("\nTime to allocate resources and prepare data:\t\t\t%.4fms", (double)(tAllocationAndDataPreparation - tStart)/((double)CLOCKS_PER_SEC/1000));
    printf("\nTime to execute axpyi using GPU:\t\t\t\t%.4fms", (double)(tAxpyi - tAllocationAndDataPreparation)/((double)CLOCKS_PER_SEC/1000));
    printf("\nTime to free resources:\t\t\t\t\t\t%.4fms", (double)(tFreeMemory - tAxpyi)/((double)CLOCKS_PER_SEC/1000));
    printf("\nTime to execute axpyi using CPU:\t\t\t\t%.4fms", (double)(tAxpyiSequential - tFreeMemory)/((double)CLOCKS_PER_SEC/1000));
    printf("\nTime to check results:\t\t\t\t\t\t%.4fms", (double)(tCheck - tAxpyiSequential)/((double)CLOCKS_PER_SEC/1000));
    printf("\nTime total:\t\t\t\t\t\t\t%.4fms", (double)(tCheck - tStart)/((double)CLOCKS_PER_SEC/1000));
    printf("\n");
     */
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

// Function che calcola la somma di due vettori
//     input:
//         - a - primo vettore
//         - b - secondo vettore
//         - n - dimensione di entrambi i vettori
//     output:
//         - c - risultato ottenuto sommando componente per componente i due vettori
void sommaArrayCompPerCompCPU(double * a, double * b, int n, double * c)
{
    int i;
    for(i=0; i<n; i++)
        c[i]=a[i]+b[i];
}

// Function che calcola il prodotto tra un vettore e uno scalare
//     input:
//         - a - primo vettore
//         - b - scalare
//         - n - dimensione del vettore
//     output:
//         - ris - vettore risultato in cui l'i-esimo componente è il prodotto dell'i-esimo componente di a per b
void prodottoScalareCPU(double * a, double b, int n, double * ris)
{
    int i;
    for(i=0;i<n;i++)
        ris[i] = a[i] * b;
}

// Function che calcola il risultato dell'operazione axpyi -> [y = y + alpha * x]
//     input:
//         - x - array sparsp
//         - n - dimensione del vettore
//         - alpha - scalare
//     output:
//         - y - vettore risultato in cui l'i-esimo componente è la somma dell'i-esimo componente del vettore stesso e il prodotto tra alpha e l'i-esimo componente del vettore x
void axpyiSequential(double * x, int n, double alpha, double * y) {

	// Variabili di appoggio
	double *initialVec;
	double *prodScalVec;

	// Allocazione memoria
	initialVec = (double *)malloc(n*sizeof(double));
	prodScalVec = (double *)malloc(n*sizeof(double));

	// Inizializzazione variabili di appoggio
	copyArray(y, initialVec, n);

	// Prodotto scalare, seconda parte dell'operazione -> [alpha * x]
	prodottoScalareCPU(x, alpha, n, prodScalVec);

	// Somma, prima parte dell'operazione -> [y = y + prodScalVec]
	sommaArrayCompPerCompCPU(initialVec, prodScalVec, n, y);

	// Libera la memoria
	free(initialVec);
	free(prodScalVec);
}