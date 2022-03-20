#include <cuda_runtime.h>

void printHArray(int hArray[], int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printHArrayF(float hArray[], int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%f", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printHArrayD(double hArray[], int size) {
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%f", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printArray(int* dArray, int size) {
    int hArray[size];
    cudaMemcpy(hArray, dArray, size * sizeof(int), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%d", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printArrayF(float* dArray, int size) {
    float hArray[size];
    cudaMemcpy(hArray, dArray, size * sizeof(float), cudaMemcpyDeviceToHost);
    printHArrayF(hArray, size);
}

void printArrayD(double* dArray, int size) {
    double hArray[size];
    cudaMemcpy(hArray, dArray, size * sizeof(double), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%f", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}

void printArrayC(cuComplex* dArray, int size) {
    cuComplex hArray[size];
    cudaMemcpy(hArray, dArray, size * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%f", hArray[i].x);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}