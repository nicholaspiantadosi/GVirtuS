#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "utilities2.h"

// Interfacce che usano funcion CUSPARSE
void mat2csr(cusparseHandle_t, cusparseMatDescr_t, double *, int, int, int, double *, int *, int *);
void bsrnnz(cusparseHandle_t, cusparseMatDescr_t, int *, int *, int, int, int, int, int, int, int *, int &);

// Interfacce per calcoli su host
void sommaArrayCompPerCompCPU(double *, double *, int, double *);
void prodottoScalareCPU(double *, double, int, double *);
void prodottoScalareArrayCPU(double *, double *, int, double, double *);
void prodottoMatriceVettoreCPU(double *, double *, int, int, double, double *);
void aAxpbySequential(double *, double *, int, int, double, double, double *);

int main(int argn, char *argv[])
{
	// Inizio esecuzione
	clock_t tStart = clock();

	// Variabili generiche
	int m, n, nnz, blockDim, mb=0, nb=0, nnzb=0, csrOperation;
	bool flag_print = true;

    // Variabili su host
    double *matrix_host;
	// double *csr_values_result;
	// int *csr_columns_result, *csr_offsets_result;
	// double *bsr_values_result;
	// int *bsr_columns_result, *bsr_offsets_result;
	double *x_host, *y_result_csr, *y_result_bsr;
	double alpha, beta;
	double *matrix_host_sequential;
	double *y_result_sequential;

	// Variabili su device
	double *csrValA_device;
	int *csrRowPtrA_device, *csrColIndA_device;
	double *bsrValC_device;
	int *bsrRowPtrC_device, *bsrColIndC_device;
	double *x_device, *y_result_csr_device, *y_result_bsr_device;

	// Imposto il seed in base all'orario
	srand((unsigned)time(0));

	printf("\n");
	printf("################################################################################################################################\n");
    printf("##########            Calcolo del prodotto di una matrice sparsa per vettore mediante la libreria CUSPARSE            ##########\n"); 
	printf("################################################################################################################################\n\n");

	printf("Se NON vengono forniti dati in input uso valori default (il vettore e la matrice avranno elementi con valori casuali):\n");

    // Controlli su input
    if(argn < 8)
	{ 
		printf("Numero di parametri insufficiente --> Uso dei valori di default\n");
		m = 4;
		n = 5;
        nnz = 9;
		blockDim = 2;
		alpha = 3;
		beta = 2;
		csrOperation = 1;
	}
	else
	{
		printf("Numero di parametri corretto\n");
		m = atoi(argv[1]);
		n = atoi(argv[2]);
        nnz = atoi(argv[3]);
		blockDim = atoi(argv[4]);
		alpha = atoi(argv[5]);
		beta = atoi(argv[6]);
		csrOperation = atoi(argv[7]);
	}

	printf("\n");
	printf("********************************************************************************************************************************\n");
	if (csrOperation == 1)
		printf("**********                                       OPERAZIONE SELEZIONATA --> CSR                                       **********\n"); 
	else
		printf("**********                                       OPERAZIONE SELEZIONATA --> BSR                                       **********\n"); 
	printf("********************************************************************************************************************************\n\n");

	printf("\tM=%d righe\n", m);
	printf("\tN=%d colonne\n", n);
	printf("\tnnz=%d valori diversi da 0\n", nnz);
	printf("\tblockDim=%d dimensione blocco per formato BSR\n", blockDim);
	printf("\talpha=%f per operazione bsrmv\n", alpha);
	printf("\tbeta=%f per operazione bsrmv\n", beta);
	printf("\tcsrOperation=%d per effettuare operazione csrmv (0 per bsrmv, 1 per csrmv)\n", csrOperation);
	if (beta > 0) {
		printf("\tè stato indicato beta > 0, l'array di output y verrà inizializzato con valori random\n");

	}
	printf("\n");

	// Controllo su numero di valori diversi da zero richiesti
    if (nnz > (m * n / 2)) {
		printf("Per rappresentare la matrice in un formato sparso, i valori devono essere al massimo pari alla metà della dimensione.\n");
		return 2;
	}

	// Calcolo mb e nb a partire da blockDim
	mb = (m + blockDim - 1) / blockDim;
	nb = (n + blockDim - 1) / blockDim;

    // Allocazione memoria sull'host
	matrix_host = (double *)malloc((m*n+1)*sizeof(double));
	// csr_values_result=(double *)malloc(nnz * sizeof(double));
	// csr_offsets_result=(int *)malloc((m + 1) * sizeof(int));
	// csr_columns_result=(int *)malloc(nnz * sizeof(int));
	x_host = (double *)malloc(n*sizeof(double));
	if (csrOperation == 1)
		y_result_csr = (double *)malloc(m*sizeof(double));
	else
		y_result_bsr = (double *)malloc(m*sizeof(double));
	matrix_host_sequential = (double *)malloc((m*n+1)*sizeof(double));
	y_result_sequential = (double *)malloc(m*sizeof(double));

	// Allocazione memoria su device
	CHECK_CUDA(cudaMalloc((void**) &x_device, n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &y_result_csr_device, m * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &y_result_bsr_device, (mb*blockDim) * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &csrValA_device, nnz * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &csrRowPtrA_device, (m + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void**) &csrColIndA_device, nnz * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void**) &bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &bsrRowPtrC_device, (mb + 1) * sizeof(int)));
	CHECK_CUDA(cudaMalloc((void**) &bsrColIndC_device, nnzb * sizeof(int)));

	// Inizializzazione variabili sull'host
    initializeMatrixRandomSparse(matrix_host, m, n, nnz);
	initializeArrayRandom(x_host, n);
	if (beta > 0)
		if (csrOperation == 1)
			initializeArrayRandom(y_result_csr, m);
		else
			initializeArrayRandom(y_result_bsr, m);
	else
		if (csrOperation == 1)
			initializeArrayToZero(y_result_csr, m);
		else
			initializeArrayToZero(y_result_bsr, m);
	
	// Copia variabili su device
	CHECK_CUDA(cudaMemcpy(x_device, x_host, n * sizeof(double), cudaMemcpyHostToDevice));
	if (csrOperation == 1)
	{
		copyArray(y_result_csr, y_result_sequential, m);
		CHECK_CUDA(cudaMemcpy(y_result_csr_device, y_result_csr, m * sizeof(double), cudaMemcpyHostToDevice));
	}
	else
	{
		copyArray(y_result_bsr, y_result_sequential, m);
		CHECK_CUDA(cudaMemcpy(y_result_bsr_device, y_result_bsr, m * sizeof(double), cudaMemcpyHostToDevice));
	}

	// Swap formato matrice per calcolo sequenziale su host
	swapMatrix(matrix_host, m, n, matrix_host_sequential);

    // Stampa variabili generate
	if (m <= 40 && n <= 40)
	{
		printf("Matrice sparsa:\n");
		stampaMatrix(matrix_host, m, n);
		printf("\n");

		printf("Array x:\n");
		stampaArrayF(x_host, n);
		printf("\n");

		printf("Matrice in formato denso per calcolo sequenziale:\n");
		stampaMatrixF1(matrix_host_sequential, m, n);
		printf("\n");

		if (beta > 0) {
			printf("Array y:\n");
			stampaArrayF(y_result_csr, m);
			printf("\n");
		}
	}	
	else
    {
		printf("Numero eccessivo di valori, la matrice non verrà visualizzata a video...\n\n");
		flag_print = false;
    }

	// Dichiarazione dell'handle per CUSPARSE
	cusparseHandle_t handle;

	// Creazione dell'handle per CUSPARSE
	CHECK_CUSPARSE(cusparseCreate(&handle));

	// Creazione della struttura della matrice con relative proprietà e forma
	cusparseMatDescr_t descr = 0;
	cusparseCreateMatDescr(&descr);
	cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

	// Tempo inizializzazioni
	clock_t tInit = clock();

	// Conversione matrice in formato CSR
	mat2csr(handle, descr, matrix_host, m, n, nnz, csrValA_device, csrRowPtrA_device, csrColIndA_device);
	printf("Matrice sparsa convertita in formato CSR\n\n");

	if (csrOperation == 1) 
	{
		// Operazione y = alpha * A * x + beta * y
		// la matrice A è rappresentata in formato CSR dagli array csrValA_device, csrRowPtrA_device e csrColIndA_device
		CHECK_CUSPARSE(cusparseDcsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, &alpha, descr, csrValA_device, csrRowPtrA_device, csrColIndA_device, x_device, &beta, y_result_csr_device));

		// Copia valori da device a host
		CHECK_CUDA(cudaMemcpy(y_result_csr, y_result_csr_device, m * sizeof(double), cudaMemcpyDeviceToHost));

		if (flag_print) 
		{
			//Stampa array risultato dall'operazione vsrmv tra matrice in formato CSR, vettore x_host, alpha e beta
			printf("Vettore risultato dall'operazione csrmv\n");
			stampaArrayF(y_result_csr, m);
			printf("\n");
		}
	}
	else
	{
		// Allocazione variabili su device
		CHECK_CUDA(cudaMalloc((void**)&bsrRowPtrC_device, (mb + 1) * sizeof(int)));

		// Calcolo bsrRowPtrC e blocchi diversi da zero
		bsrnnz(handle, descr, csrRowPtrA_device, csrColIndA_device, m, n, nnz, blockDim, mb, nb, bsrRowPtrC_device, nnzb);

		// Allocazione variabili su host sulla base del numero di blocchi diversi da zero
		CHECK_CUDA(cudaMalloc((void**)&bsrColIndC_device, nnzb * sizeof(int)));
		CHECK_CUDA(cudaMalloc((void**)&bsrValC_device, (blockDim * blockDim) * nnzb * sizeof(double)));

		// Conversione da CSR a BSR
		CHECK_CUSPARSE(cusparseDcsr2bsr(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrValA_device, csrRowPtrA_device, csrColIndA_device, blockDim, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device));
		// Stampa matrice convertita in formato BSR
		printf("Matrice sparsa convertita in formato BSR\n");

		// Operazione y = alpha * A * x + beta * y
		// la matrice A è rappresentata in formato BSR dagli array bsrValC_device, bsrRowPtrC_device e bsrColIndC_device
		CHECK_CUSPARSE(cusparseDbsrmv(handle, CUSPARSE_DIRECTION_COLUMN, CUSPARSE_OPERATION_NON_TRANSPOSE, mb, nb, nnzb, &alpha, descr, bsrValC_device, bsrRowPtrC_device, bsrColIndC_device, blockDim, x_device, &beta, y_result_bsr_device));

		// Copia valori da device a host
		CHECK_CUDA(cudaMemcpy(y_result_bsr, y_result_bsr_device, m * sizeof(double), cudaMemcpyDeviceToHost));

		if (flag_print) 
		{
			//Stampa array risultato dall'operazione bsrmv tra matrice in formato BSR, vettore x_host, alpha e beta
			printf("Vettore risultato dall'operazione bsrmv\n");
			stampaArrayF(y_result_bsr, m);
			printf("\n");
		}
	}

	// Tempo operazioni CUSPARSE
	clock_t tCusparseOperations = clock();

	// Operazione y = alpha * A * x + beta * y effettuata in modalità sequenziale
	aAxpbySequential(matrix_host_sequential, x_host, m, n, alpha, beta, y_result_sequential);

	// Tempo operazione sequenziale
	clock_t tSequentialOperation = clock();

	if (flag_print) 
	{
		//Stampa array risultato dall'operazione [y = alpha * A * x + beta * y] sequenziale tra matrice in formato denso, vettore x_host, alpha e beta
		printf("Vettore risultato dall'operazione [y = alpha * A * x + beta * y] sequenziale\n");
		stampaArrayF(y_result_sequential, m);
		printf("\n");
	}

	// Verifica se i due array risultato (uno calcolato con cusparse sul device e uno calcolato su host) coincidono
	if (csrOperation == 1)
		equalArrayD(y_result_csr, y_result_sequential, m);
	else
		equalArrayD(y_result_bsr, y_result_sequential, m);

	//Libera la memoria sull'host
	if (csrOperation == 1) {
		// free(csr_values_result);
		// free(csr_offsets_result);
		// free(csr_columns_result);
	} else {
		// free(bsr_values_result);
		// free(bsr_offsets_result);
		// free(bsr_columns_result);
	}
	free(matrix_host);
	free(matrix_host_sequential);
	// free(x_host);
	// free(y_result_csr);
	// free(y_result_bsr);
	// free(y_result_sequential);

	// Libera la memoria su device
	CHECK_CUDA(cudaFree(csrValA_device));
	CHECK_CUDA(cudaFree(csrRowPtrA_device));
	CHECK_CUDA(cudaFree(csrColIndA_device));
	CHECK_CUDA(cudaFree(bsrValC_device));
	CHECK_CUDA(cudaFree(bsrRowPtrC_device));
	CHECK_CUDA(cudaFree(bsrColIndC_device));
	CHECK_CUDA(cudaFree(x_device));
	CHECK_CUDA(cudaFree(y_result_csr_device));
	CHECK_CUDA(cudaFree(y_result_bsr_device));

	// Termina l'handle per CUSPARSE
	CHECK_CUSPARSE(cusparseDestroyMatDescr(descr));
    CHECK_CUSPARSE(cusparseDestroy(handle));

	// Tempo liberazione memoria
	clock_t tFinish = clock();

	// Log tempi esecuzione
	printf("\nTempo inizializzazione:\t\t%fms", (double)(tInit - tStart)/(CLOCKS_PER_SEC/1000));
	printf("\nTempo operazioni cusparse:\t%fms", (double)(tCusparseOperations - tInit)/(CLOCKS_PER_SEC/1000));
	printf("\nTempo operazione sequenziale:\t%fms", (double)(tSequentialOperation - tCusparseOperations)/(CLOCKS_PER_SEC/1000));
	printf("\nTempo liberazione memoria:\t%fms", (double)(tFinish - tSequentialOperation)/(CLOCKS_PER_SEC/1000));
	printf("\nTempo totale:\t\t\t%fms\n\n", (double)(tFinish - tStart)/(CLOCKS_PER_SEC/1000));
}

// Function che consente di ottenere la matrice in formato CSR
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - matrix - matrice sparsa
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//     output:
//         - csrValA - array contenente i valori
//         - csrRowPtrA - array contenente gli indici per i quali bisogna considerare il nuovo indice di riga
//         - csrColIndA - array contenente gli indici di colonne
void mat2csr(cusparseHandle_t handle, cusparseMatDescr_t descr, double * matrix, int m, int n, int nnz, double * csrValA, int * csrRowPtrA, int * csrColIndA)
{
	int nnz_total = 0;

	// Variabili su device
	double *matrix_device;
	int *nnz_per_row;

	// Allocazione memoria su device
	CHECK_CUDA(cudaMalloc((void**) &matrix_device, m * n * sizeof(double)));
	CHECK_CUDA(cudaMalloc((void**) &nnz_per_row, m * sizeof(int)));

	// Inizializzazione variabili su device
	CHECK_CUDA(cudaMemcpy(matrix_device, matrix, m * n * sizeof(double), cudaMemcpyHostToDevice));

	// Calcolo valori diversi da zero
	CHECK_CUSPARSE(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, m, n, descr, matrix_device, m, nnz_per_row, &nnz_total));

	// Controllo su valori diversi da zero richiesti in input rispetto a quelli calcolati tramite cusparseSnnz()
	if (nnz != nnz_total) {
		printf("I valori diversi da zero richiesti in input sono diversi rispetto a quelli rilevati: richiesti %d valori ma sono stati rilevati %d valori diversi da zero!\n\n", nnz, nnz_total);
		exit(EXIT_FAILURE);
	}

	// Conversione matrice in formato CSR
	CHECK_CUSPARSE(cusparseDdense2csr(handle, m, n, descr, matrix_device, m, nnz_per_row, csrValA, csrRowPtrA, csrColIndA));


	//Libera la memoria sul device
    CHECK_CUDA(cudaFree(matrix_device));
	CHECK_CUDA(cudaFree(nnz_per_row));
}

// Function che consente di ottenere il numero dei blocchi con elementi diversi da zero e l'array bsrRowPtrC inizializzato con i valori di indice per i blocchi di riga
//     input:
//         - handle - handle di CUSPARSE
//         - descr - proprietà e forma della matrice
//         - csrRowPtrA - array contenente gli indici per i quali bisogna considerare il nuovo indice di riga, della matrice in formato CSR
//         - csrColIndA - array contenente gli indici di colonne, della matrice in formato CSR
//         - m - numero di righe
//         - n - numero di colonne
//         - nnz - numero di valori diversi da zero
//         - blockDim - dimensione che deve avere il blocco nel formato BSR
//         - mb - numero di righe dei blocchi calcolati come segue: mb = (m + blockDim - 1) / blockDim
//         - nb - numero di colonne dei blocchi calcolati come segue: nb = (n + blockDim - 1) / blockDim;
//     output:
//         - bsrRowPtrC - array contenente i valori di indice per i blocchi di riga della matrice in formato BSR
//         - nnzb - blocchi con valori diversi da zero
void bsrnnz(cusparseHandle_t handle, cusparseMatDescr_t descr, int * csrRowPtrA, int * csrColIndA, int m, int n, int nnz, int blockDim, int mb, int nb, int * bsrRowPtrC, int &nnzb)
{
	// Variabili su host
	int base;
	int *nnzTotalBsr = &nnzb;
	
	// Calcolo del numero dei blocchi diversi da zero per il formato BSR
	CHECK_CUSPARSE(cusparseXcsr2bsrNnz(handle, CUSPARSE_DIRECTION_COLUMN, m, n, descr, csrRowPtrA, csrColIndA, blockDim, descr, bsrRowPtrC, nnzTotalBsr));

	// Controllo sul valore dei blocchi calcolato, se null lo calcolo sulla base degli indici dei blocchi e il numero dei blocchi per riga
	if (NULL != nnzTotalBsr)
	{
		nnzb = *nnzTotalBsr;
	}
	else
	{
		CHECK_CUDA(cudaMemcpy(&nnzb, bsrRowPtrC + mb, sizeof(int), cudaMemcpyDeviceToHost));
		CHECK_CUDA(cudaMemcpy(&base, bsrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost));
		nnzb -= base;
	}

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

// Function che calcola il prodotto scalare tra due vettori
//     input:
//         - a - primo vettore
//         - b - secondo vettore
//         - n - dimensione di entrambi i vettori
//         - alpha - se > -1 il risultato viene moltiplicato per alpha
//     output:
//         - ris - risultato ottenuto dalla sommatoria dei prodotti dei singoli elementi dei vettori, moltiplicato per alpha (se > -1)
void prodottoScalareArrayCPU(double * a, double * b, int n, double alpha, double * ris)
{
    int i;
    for(i=0;i<n;i++)
        *ris+=(a[i]*b[i]);
	if (alpha > -1)
		*ris*=alpha;
}

// Function che calcola il prodotto di una matrice (dimensione m x n) per un vettore (dimensione n)
//     input:
//         - matrix - matrice
//         - vector - vettore
//         - m - numero di righe della matrice
//         - n - numero di colonne della matrice corrispondente con la dimensione del vettore
//         - alpha - valore per cui deve essere moltiplicato ogni elemento di out
//     output:
//         - out - vettore risultato di dimensione m in cui l'elemento i contiene il prodotto dell'i-esima riga della matrice e il vettore
void prodottoMatriceVettoreCPU(double * matrix, double * vector, int m, int n, double alpha, double * out)
{
    for(int i=0; i<m;i++)
        prodottoScalareArrayCPU(&matrix[i*n], vector, n, alpha, &out[i]);
}

// Function che calcola il prodotto [y = alpha * A * x + beta * y] in modalità sequenziale
//     input:
//         - matrix - matrice
//         - vector - vettore
//         - m - numero di righe della matrice
//         - n - numero di colonne della matrice corrispondente con la dimensione del vettore
//         - alpha - valore per cui deve essere moltiplicato la prima parte dell'operazione
//         - beta - valore per cui deve essere moltiplicato la seconda parte dell'operazione
//     output:
//         - out - vettore risultato di dimensione m in cui l'elemento è la somma dell'elemento stesso per beta, più il prodotto tra la matrice e il vettore scalato di un fattore alpha
void aAxpbySequential(double * matrix, double * vector, int m, int n, double alpha, double beta, double * out)
{
	// Dichiarazione variabili di appoggio su host
	double *out1, *out2;

	// Allocazione su host
	out1 = (double *)malloc(m*sizeof(double));
	out2 = (double *)malloc(m*sizeof(double));

	// Inizializzazione a 0 di tutti i vettori
	initializeArrayToZero(out1, m);
	initializeArrayToZero(out2, m);

	// Prima parte dell'operazione [alpha * A * x]
	prodottoMatriceVettoreCPU(matrix, vector, m, n, alpha, out1);

	// Seconda parte dell'operazione [beta * y]
	prodottoScalareCPU(out, beta, m, out2);

	// Somma dei vettori risultanti delle due precedenti operazioni
	sommaArrayCompPerCompCPU(out1, out2, m, out);

	// Libera memoria
	free(out1);
	free(out2);
}
