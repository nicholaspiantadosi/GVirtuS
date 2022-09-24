#include<cuda.h>
#include<stdio.h>
#include <iostream>
#include <chrono>
typedef std::chrono::high_resolution_clock Clock;

void initializeArray(double*,int);
void initializeArray(double *, int, int);
void initializeArrayToZero(double *, int);
void initializeArrayRandom(double *, int);
void initializeArrayRandomSparse(double *, int, int);
void stampaArray(double*, int);
void equalArray(double*, double*, int);
void prodottoScalareArrayCPU(double *, double *, int, double );
void sommaArrayCompPerCompCPU(double *, double *, int);
__global__ void prodottoScalareArrayGPU(double*, double*, int, double );
__global__ void sommaArrayCompPerCompGPU(double*, double*, int );
void copyArray(double *, double *, int);

void execute(int, int, double, int, dim3);

int main(int argn, char *argv[])
{
    std::cout << "n;blocks;threads;allocation (micros);execution (micros);free (micros);total (micros);" << std::endl;

    /*
    execute(10, 1, 2, 1, 1);
    execute(10, 1, 2, 1, 2);
    execute(10, 1, 2, 1, 5);
    execute(10, 1, 2, 1, 10);

    execute(100, 1, 2, 1, 1);
    execute(100, 1, 2, 1, 2);
    execute(100, 1, 2, 1, 5);
    execute(100, 1, 2, 1, 10);

    execute(1000, 100, 2, 1, 1);
    execute(1000, 100, 2, 1, 2);
    execute(1000, 100, 2, 1, 5);
    execute(1000, 100, 2, 1, 10);

    execute(10000, 1000, 2, 1, 1);
    execute(10000, 1000, 2, 1, 2);
    execute(10000, 1000, 2, 1, 5);
    execute(10000, 1000, 2, 1, 10);

    execute(100000, 10000, 2, 1, 1);
    execute(100000, 10000, 2, 1, 2);
    execute(100000, 10000, 2, 1, 5);
    execute(100000, 10000, 2, 1, 10);

    execute(1000000, 100000, 2, 1, 1);
    execute(1000000, 100000, 2, 1, 2);
    execute(1000000, 100000, 2, 1, 5);
    execute(1000000, 100000, 2, 1, 10);

    execute(10000000, 1000000, 2, 1, 1);
    execute(10000000, 1000000, 2, 1, 2);
    execute(10000000, 1000000, 2, 1, 5);
    execute(10000000, 1000000, 2, 1, 10);

    execute(100000000, 10000000, 2, 1, 1);
    execute(100000000, 10000000, 2, 1, 2);
    execute(100000000, 10000000, 2, 1, 5);
    execute(100000000, 10000000, 2, 1, 10);

    execute(1000000000, 100000000, 2, 1, 1);
    execute(1000000000, 100000000, 2, 1, 2);
    execute(1000000000, 100000000, 2, 1, 5);
    execute(1000000000, 100000000, 2, 1, 10);
    */
/*
    execute(10, 1, 2, 1, 4);
    execute(100, 10, 2, 1, 4);
    execute(1000, 100, 2, 1, 4);
    execute(10000, 1000, 2, 1, 4);
    execute(100000, 10000, 2, 1, 4);
    execute(1000000, 100000, 2, 1, 4);
    execute(10000000, 1000000, 2, 1, 4);
    execute(100000000, 10000000, 2, 1, 4);
    execute(1000000000, 100000000, 2, 1, 4);
*/
    execute(10, 0, 2, 1, 4);
    execute(100, 1, 2, 1, 4);
    execute(1000, 10, 2, 1, 4);
    execute(10000, 100, 2, 1, 4);
    execute(100000, 1000, 2, 1, 4);
    execute(1000000, 10000, 2, 1, 4);
    execute(10000000, 100000, 2, 1, 4);
    execute(100000000, 1000000, 2, 1, 4);
    execute(1000000000, 10000000, 2, 1, 4);
}

void execute(int n, int nnz, double alpha, int yRandom, dim3 nThreadPerBlocco) {
    auto t1 = Clock::now();

    dim3 nBlocchi;
    double *vector_host, *y_result_host, *y_temp_host;
    double *x_device, *y_device, *y_temp_device;
    double *copy;
    double size;

    nBlocchi=n/nThreadPerBlocco.x+((n%nThreadPerBlocco.x)==0? 0:1);
    size=n*sizeof(double);
    vector_host=(double*)malloc(size);
    y_result_host=(double*)malloc(size);
    y_temp_host=(double*)malloc(size);
    copy=(double*)malloc(size);
    cudaMalloc((void**)&x_device, size);
    cudaMalloc((void**)&y_device, size);
    cudaMalloc((void**)&y_temp_device, size);

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
    copyArray(y_result_host, y_temp_host, n);

    cudaMemcpy(x_device, vector_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_device, y_result_host, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_temp_device, y_temp_host, size, cudaMemcpyHostToDevice);

    auto t2 = Clock::now();

    prodottoScalareArrayGPU<<<nBlocchi, nThreadPerBlocco>>>(x_device, y_temp_device, n, alpha);

    //auto t2_2 = Clock::now();
    //std::cout << "prodottoScalareArrayGPU: " << std::chrono::duration_cast<std::chrono::microseconds>(t2_2 - t2).count() << std::endl;

    sommaArrayCompPerCompGPU<<<nBlocchi, nThreadPerBlocco>>>(y_temp_device, y_device, n);

    //auto t2_3 = Clock::now();
    //std::cout << "sommaArrayCompPerCompGPU: " << std::chrono::duration_cast<std::chrono::microseconds>(t2_3 - t2_2).count() << std::endl;

    cudaMemcpy(copy, y_device, size, cudaMemcpyDeviceToHost);

    auto t3 = Clock::now();

    //std::cout << "copied: " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2_3).count() << std::endl;

    /*
    prodottoScalareArrayCPU(vector_host, y_temp_host, n, alpha);
    sommaArrayCompPerCompCPU(y_temp_host, y_result_host, n);

    equalArray(copy, y_result_host, n);

    printf("y_result_host\n");
    stampaArray(y_result_host, n);
    printf("Risultati device\n");
    stampaArray(copy, n);
    */

    free(vector_host);
    free(y_result_host);
    free(copy);
    cudaFree(x_device);
    cudaFree(y_device);

    auto t4 = Clock::now();

    //std::cout << "free: " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << std::endl;

    // Fine esecuzione
    /*
    std::cout << "axpyi - y = y + alpha * x" << std::endl;
    std::cout << "alpha = " << alpha << std::endl;
    std::cout << "n = " << n << std::endl;
    std::cout << "nnz = " << nnz << std::endl;
    std::cout << "Time to allocate resources and prepare data: " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << std::endl;
    std::cout << "Time to execute axpyi using GPU (pre CUDA):  " << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << " microseconds" << std::endl;
    std::cout << "Time to free resources:                      " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << std::endl;
    std::cout << "Time total:                                  " << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << " microseconds" << std::endl;
     */
    std::cout << n << ";" << nBlocchi.x << ";" << nThreadPerBlocco.x << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << ";" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t1).count() << ";" << std::endl;
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

void initializeArrayToZero(double *array, int n)
{
    int i;
    for(i=0;i<n;i++)
        array[i] = 0;
}

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

void stampaArray(double* array, int n)
{
    int i;
    for(i=0;i<n;i++)
        printf("%f ", array[i]);
    printf("\n");
}

void equalArray(double* a, double*b, int n)
{
    int i=0;
    while(a[i]==b[i])
        i++;
    if(i<n)
        printf("I risultati dell'host e del device sono diversi\n");
    else
        printf("I risultati dell'host e del device coincidono\n");
}

// Seriale
void prodottoScalareArrayCPU(double *a, double *b, int n, double k)
{
    int i;
    for(i=0; i<n; i++)
        b[i]=a[i]*k;
}

//parallelo
__global__ void prodottoScalareArrayGPU(double *a, double *b, int n, double k)
{
    int index=threadIdx.x + blockIdx.x*blockDim.x;
    if (index < n)
        b[index] = a[index]*k;
}

// Seriale
void sommaArrayCompPerCompCPU(double *x, double *y, int n)
{
    int i;
    for(i=0; i<n; i++)
        y[i]=x[i]+y[i];
}

//parallelo
__global__ void sommaArrayCompPerCompGPU(double *x, double *y, int n)
{
    int index=threadIdx.x + blockIdx.x*blockDim.x;
    if (index < n)
        y[index] = x[index]+y[index];
}

void copyArray(double * source, double * destination, int n) {
    int i;
    for(i=0;i<n;i++)
        destination[i] = source[i];
}