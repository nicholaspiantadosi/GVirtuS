#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

// Interfacce per calcoli su host
void sommaArrayCompPerCompCPU(double *, double *, int, double *);
void prodottoScalareCPU(double *, double, int, double *);
void axpyiSequential(double *, int, double, double *);

void stampaArray(double*, int);
void initializeArray(double *, int);
void initializeArray(double *, int, int);
void initializeArrayRandom(double *, int);
void initializeArrayToZero(double *, int);
void initializeArrayRandomSparse(double *, int, int);
void copyArray(double *, double *, int);

void execute(int, int, double, int);

int main(int argn, char *argv[])
{
    // Variabili generiche
    //int nnz;
    //int yRandom;
    //double alpha;

    //n = atoi(argv[1]);
    //nnz = atoi(argv[2]);
    //alpha = atof(argv[3]);
    //yRandom = atoi(argv[4]);

    //printf("n: %d; nnz = %d, alpha = %f, yRandom = %d\n", n, nnz, alpha, yRandom);

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
    execute(100000000, 1000000, 2, 1);
    execute(1000000000, 10000000, 2, 1);

}

void execute(int n, int nnz, double alpha, int yRandom) {
    // Inizio esecuzione
    auto t1 = Clock::now();



    // Variabili su host
    double *vector_host;
    double *y_result_host;

    // Imposto il seed in base all'orario
    srand(time(0));

    // Allocazione memoria sull'host
    vector_host = (double *)malloc(n*sizeof(double));
    y_result_host = (double *)malloc(n*sizeof(double));

    // Inizializzazione variabili sull'host
    /*
    initializeArray(vector_host, n);
    initializeArrayToZero(y_result_host, n);
    if(yRandom > 0) {
        initializeArray(y_result_host, n, yRandom);
    }
     */
    initializeArrayRandomSparse(vector_host, n, nnz);
    initializeArrayToZero(y_result_host, n);
    if(yRandom > 0) {
        initializeArrayRandom(y_result_host, n);
    }

    auto t2 = Clock::now();

    // Operazione axpyi in sequenziale
    axpyiSequential(vector_host, n, alpha, y_result_host);

    /*
    printf("y_result_host\n");
    stampaArray(y_result_host, n);
    */

    auto t3 = Clock::now();

    //Libera la memoria sull'host
    free(vector_host);
    free(y_result_host);

    auto t4 = Clock::now();

    // Fine esecuzione
    /*
    std::cout << "axpyi - y = y + alpha * x" << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << "Time to allocate resources and prepare data: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << std::endl;
    std::cout << "Time to execute axpyi using CPU:             " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " microseconds" << std::endl;
    std::cout << "Time to free resources:                      " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << std::endl;
    std::cout << "Time total:                                  " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << " microseconds" << std::endl;
     */
    std::cout << n << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << ";" << std::endl;
}

void stampaArray(double* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
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

// Function che inizializza un vettore di tipo intero
//     input:
//         - array - vettore
//         - n - dimensione dell'array
//     output:
//         - array - vettore inizializzato
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