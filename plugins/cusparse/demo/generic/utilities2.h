#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

// Function per la stampa di un vettore di int
//     input:
//         - array - vettore
//         - n - dimensione del vettore
//     output:
//         - stampa a video dell'array di dimensione n
void stampaArray(int* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%d ", array[i]);
    printf("\n");
}

// Function per la stampa di un vettore di double
//     input:
//         - array - vettore
//         - n - dimensione del vettore
//     output:
//         - stampa a video dell'array di dimensione n
void stampaArrayF(double* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
}

// Function per il confronto di due vettori
//     input:
//         - a - primo vettore
//         - b - secondo vettore
//         - n - dimensione di entrambi i vettori
//     output:
//         - stampa a video del risultato del confronto elemento per elemento degli array a e b
void equalArray(int* a, int*b, int n)
{
    int i=0;
    while(a[i]==b[i])
        i++;
    if(i<n)
        printf("I risultati dell'host e del device sono diversi\n");
    else
        printf("I risultati dell'host e del device coincidono\n");
}

// Function per la copia degli elementi da un vettore ad un altro
//     input:
//         - source - vettore di input
//         - n - dimensione di entrambi i vettori
//     output:
//         - destination - vettore di output contenente gli stessi elementi dell'array source
void copyArray(double * source, double * destination, int n) {
	int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}

// Function che verifica se i due input sono uguali
//     input:
//         - x - primo elemento di tipo float
//         - y - secondo elemento di tipo float
//         - epsilon - se non fornito in input viene assegnato il valore di default 0.1f
//     output:
//         - bool - true se i valori coincidono, false se non coincidono
bool compare_float(float x, float y, float epsilon = 0.1f){
   if(fabs(x - y) < epsilon)
      return true; //they are same
      return false; //they are not same
}

// Function che verifica se due vettori sono uguali
//     input:
//         - a - primo vettore
//         - b - secondo vettore
//         - n - dimensione di entrambi gli array
//     output:
//         - void - stampa se i due array coincidono
void equalArrayF(float * a, float * b, int n)
{
    int i=0;
	while(compare_float(a[i], b[i]))
        i++;
    if(i<n) {
		printf("\nDIVERSI!!!\n");
		printf("index = %d --> %f != %f \n", i, a[i], b[i]);
	}
    else
        printf("\nCOINCIDONO!!!\n");
}

bool compare_double(double a, double b, double epsilon = 0.000001)
{
    return fabs(a-b) < epsilon;
}

// Function che verifica se due vettori di double sono uguali
//     input:
//         - a - primo vettore
//         - b - secondo vettore
//         - n - dimensione di entrambi gli array
//     output:
//         - void - stampa se i due array coincidono
void equalArrayD(double * a, double * b, int n)
{
    int i=0;
	//while(a[i] == b[i])
	while(compare_double(a[i], b[i]))
        i++;
    if(i<n) {
		printf("\nDIVERSI!!!\n");
		printf("index = %d --> %f != %f \n", i, a[i], b[i]);
	}
    else
        printf("\nCOINCIDONO!!!\n");
}

// Function che inizializza un vettore di tipo intero
//     input:
//         - array - vettore
//         - n - dimensione dell'array
//     output:
//         - array - vettore inizializzato
void initializeArray(int *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = i;
}

// Function che inizializza un vettore di tipo double con valori random da 1 a 100
//     input:
//         - array - vettore
//         - n - dimensione dell'array
//     output:
//         - array - vettore inizializzato
void initializeArrayRandom(double *array, int n)
{
    int i;
	double random_number;
    for(i=0;i<n;i++)
	{
		random_number=(double)rand()/((double)RAND_MAX/(100)) + 1;
		array[i] = random_number;
	}
}

// Function che inizializza un vettore di tipo double con valori pari a zero
//     input:
//         - array - vettore
//         - n - dimensione dell'array
//     output:
//         - array - vettore inizializzato
void initializeArrayToZero(double *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

// Function che inizializza una matrice sparsa di tipo double con valori random da 1 a 100 in notazione vettoriale
//     input:
//         - array - matrice in notazione vettoriale
//         - n - dimensione della matrice (corrisponde alla dimensione del vettore)
//         - nnz - numero di valori che devono essere diversi da zero
//     output:
//         - array - matrice inizializzata
void initializeArrayRandomSparse(double *array, int n, int nnz)
{
    int i;
	double random_number;
	initializeArrayToZero(array, n);
    for(i=0;i<nnz;)
	{
		int index = (int) (n * ((double) rand() / (RAND_MAX)));
		if (array[index]) { 
			continue;
		}
		random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
		array[index] = random_number;
		++i;
	}
}

// Function che inizializza una matrice di tipo double con valori random da 1 a 100
//     input:
//         - matrix - matrice
//         - M - numero di righe della matrice
//         - N - numero di colonne della matrice
//     output:
//         - matrix - matrice inizializzata
void initializeMatrixRandom(double *matrix, int M, int N)
{
	int i,j,k=0;
	double random_number;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			random_number=(double)rand()/((double)RAND_MAX/(100)) + 1;
			matrix[k++]= random_number;
		}
	}
}

// Function che inizializza una matrice di tipo double con valori pari a zero
//     input:
//         - matrix - matrice
//         - M - numero di righe della matrice
//         - N - numero di colonne della matrice
//     output:
//         - matrix - matrice inizializzata
void initializeMatrixZero(double *matrix, int M, int N)
{
	int i, j, k=0;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			matrix[k++]=0;
		}
	}
}

// Function che inizializza una matrice sparsa di tipo double con valori random da 1 a 100
//     input:
//         - matrix - matrice
//         - M - numero di righe della matrice
//         - N - numero di colonne della matrice
//         - nnz - numero di valori che devono essere diversi da zero
//     output:
//         - matrix - matrice inizializzata
void initializeMatrixRandomSparse(double *matrix, int M, int N, int nnz)
{
	initializeMatrixZero(matrix, M, N);
	int i=0;
	double random_number;
	for (i = 0; i < nnz;) {
		int index = (int) (M * N * ((double) rand() / (RAND_MAX + 1.0)));
		if (matrix[index]) { 
			continue;
		}
		random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
		matrix[index] = random_number;
		++i;
	}
}

// Function per la stampa a video di una matrice di tipo double memorizzata su righe
//     input:
//         - matrix - matrice
//         - M - numero di righe della matrice
//         - N - numero di colonne della matrice
//     output:
//         - stampa a video il contenuto della matrice
void stampaMatrixF1(double* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
            printf("%f ", matrix[i*N+j]);
        printf("\n");
    }
}

// Function per la stampa a video di una matrice di tipo double memorizzata su colonne
//     input:
//         - matrix - matrice
//         - M - numero di righe della matrice
//         - N - numero di colonne della matrice
//     output:
//         - stampa a video il contenuto della matrice
void stampaMatrix(double* matrix, int M, int N)
{
    int i,j;
    for(i=0;i<M;i++)
    {
        for(j=0;j<N;j++)
			printf("%f ", matrix[i+j*M]);
        printf("\n");
    }
}

// Function che restituisce il relativo messaggio d'errore corrispondente al codice errore CUSPARSE
//     input:
//         - error - errore ritornato da una chiamata ad una function della libreria CUSPARSE
//     output:
//         - char * - stringa contenente il messaggio d'errore relativo
const char * getErrorString(cusparseStatus_t error)
{
    switch (error)
    {
		case CUSPARSE_STATUS_SUCCESS:
			return "The operation completed successfully.";
		case CUSPARSE_STATUS_NOT_INITIALIZED:
			return "The cuSPARSE library was not initialized. This is usually caused by the lack of a prior call, an error in the CUDA Runtime API called by the cuSPARSE routine, or an error in the hardware setup.\n" \
				"To correct: call cusparseCreate() prior to the function call; and check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
	
		case CUSPARSE_STATUS_ALLOC_FAILED:
			return "Resource allocation failed inside the cuSPARSE library. This is usually caused by a cudaMalloc() failure.\n"\
					"To correct: prior to the function call, deallocate previously allocated memory as much as possible.";
	
		case CUSPARSE_STATUS_INVALID_VALUE:
			return "An unsupported value or parameter was passed to the function (a negative vector size, for example).\n"\
				"To correct: ensure that all the parameters being passed have valid values.";
	
		case CUSPARSE_STATUS_ARCH_MISMATCH:
			return "The function requires a feature absent from the device architecture; usually caused by the lack of support for atomic operations or double precision.\n"\
				"To correct: compile and run the application on a device with appropriate compute capability, which is 1.1 for 32-bit atomic operations and 1.3 for double precision.";
	
		case CUSPARSE_STATUS_MAPPING_ERROR:
			return "An access to GPU memory space failed, which is usually caused by a failure to bind a texture.\n"\
				"To correct: prior to the function call, unbind any previously bound textures.";
	
		case CUSPARSE_STATUS_EXECUTION_FAILED:
			return "The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.\n"\
					"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed.";
	
		case CUSPARSE_STATUS_INTERNAL_ERROR:
			return "An internal cuSPARSE operation failed. This error is usually caused by a cudaMemcpyAsync() failure.\n"\
					"To correct: check that the hardware, an appropriate version of the driver, and the cuSPARSE library are correctly installed. Also, check that the memory passed as a parameter to the routine is not being deallocated prior to the routine’s completion.";
	
		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			return "The matrix type is not supported by this function. This is usually caused by passing an invalid matrix descriptor to the function.\n"\
					"To correct: check that the fields in cusparseMatDescr_t descrA were set correctly.";
    }
 
    return "<unknown>";
}

// Function che verifica se lo stato CUDA è diverso da SUCCESS, in tal caso stampa a video un messaggio d'errore e esce dal programma
//     input:
//         - status - stato ritornato da una chiamata ad una function della libreria CUDA
//     output:
//         - se status == SUCCESS, continua con l'esecuzione; altrimenti stampa a video l'errore e termina l'esecuzione
void CHECK_CUDA(cudaError_t status)
{
    if (status != cudaSuccess) {
        printf("CUDA API failed at line %d with error: %s (%d)\n", __LINE__, cudaGetErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

// Function che verifica se lo stato CUSPARSE è diverso da SUCCESS, in tal caso stampa a video un messaggio d'errore e esce dal programma
//     input:
//         - status - stato ritornato da una chiamata ad una function della libreria CUSPARSE
//     output:
//         - se status == SUCCESS, continua con l'esecuzione; altrimenti stampa a video l'errore e termina l'esecuzione
void CHECK_CUSPARSE(cusparseStatus_t status)
{
    if (status != CUSPARSE_STATUS_SUCCESS) {
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n", __LINE__, getErrorString(status), status);
        exit(EXIT_FAILURE);
    }
}

// Function per la conversione di una matrice da "memorizzazione su righe" a "memorizzazione su colonne", e viceversa
//     input:
//         - matrix - matrice in input
//         - m - numero di righe della matrice
//         - n - numero di colonne della matrice
//     output:
//         - matrix_out - matrice con gli elementi della matrice in input scambiati: righe su colonne (e viceversa)
void swapMatrix(double * matrix, int m, int n, double * matrix_out)
{
	int i,j;
	for (i = 0; i < m; i++)
        for (j = 0; j < n; j++)
            matrix_out[i*n+j]= matrix[j*m+i];
}