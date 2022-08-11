#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utilities2.h"

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

	// Variabili generiche
	int n, nnz;
	bool flag_print = true;
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

	printf("\n");
	printf("################################################################################################################################\n");
    printf("##########          Calcolo del prodotto di un vettore sparso per uno scalare mediante la libreria CUSPARSE           ##########\n"); 
	printf("################################################################################################################################\n\n");

	printf("Se NON vengono forniti dati in input uso valori default (il vettore e la matrice avranno elementi con valori casuali):\n");

    // Controlli su input
    if(argn < 4)
	{ 
		printf("Numero di parametri insufficiente --> Uso dei valori di default\n");
		n = 10;
        nnz = 4;
		alpha = 2;
		yRandom = 0;
	}
	else
	{
		printf("Numero di parametri corretto\n");
		n = atoi(argv[1]);
        nnz = atoi(argv[2]);
		alpha = atoi(argv[3]);
		yRandom = atoi(argv[4]);
	}
	printf("\tN=%d\n", n);
	printf("\tnnz=%d valori diversi da 0\n", nnz);
	printf("\talpha=%f per operazione axpyi\n", alpha);
	printf("\tyRandom=%d per generare valori randomici in y\n", yRandom);
	printf("\n");

	// Controllo su numero di valori diversi da zero richiesti
    if (nnz > (n / 2)) {
		printf("Per rappresentare il vettore in un formato sparso, i valori devono essere al massimo pari alla metà della dimensione.\n");
		return 2;
	}

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

    // Stampa variabili generate
	if (n <= 40)
	{
		printf("Array x:\n");
		stampaArrayF(vector_host, n);
		printf("\n");
		if (yRandom > 0) {
			printf("Array y:\n");
			stampaArrayF(y_result, n);
			printf("\n");
		}
	}	
	else
    {
		printf("Numero eccessivo di valori, il vettore non verrà visualizzato a video...\n\n");
		flag_print = false;
    }

	// Calcolo array in formato sparso
	generateSparseVectorFormat(vector_host, n, sparse_values, sparse_col_index);

	// Stampa array in formato sparso
	if (flag_print) {
		printf("Array x in formato sparso:\n");
		printf("\txVal: ");
		stampaArrayF(sparse_values, nnz);
		printf("\n\txInd: ");
		stampaArray(sparse_col_index, nnz);
		printf("\n");
	}

	// Dichiarazione dell'handle per CUSPARSE
	cusparseHandle_t handle;

	// Creazione dell'handle per CUSPARSE
	CHECK_CUSPARSE(cusparseCreate(&handle));

	// Prodotto vettore * alpha, il risultato sommato in y (function CUSPARSE axpyi)
	axpyi(handle, n, nnz, &alpha, sparse_values, sparse_col_index, y_result);

	if (flag_print) 
	{
		//Stampa vettore risultato in formato denso
		printf("Vettore risultato:\n");
		stampaArrayF(y_result, n);
		printf("\n");
	}

	// Operazione axpyi in sequenziale
	axpyiSequential(vector_host, n, alpha, y_result_host);

	if (flag_print) 
	{
		//Stampa vettore risultato in formato denso calcolato su host in modo sequenziale
		printf("Vettore risultato calcolato in sequenziale:\n");
		stampaArrayF(y_result_host, n);
		printf("\n");
	}

	// Verifica se i due array risultato (uno calcolato con cusparse sul device e uno calcolato su host) coincidono
	equalArrayD(y_result, y_result_host, n);

	//Libera la memoria sull'host
	free(vector_host);
	free(y_result);
	free(sparse_values);
	free(sparse_col_index);
	free(y_result_host);

	// Termina l'handle per CUSPARSE
    CHECK_CUSPARSE(cusparseDestroy(handle));

	// Fine esecuzione
	printf("\nTempo totale: %.2fs\n\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
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