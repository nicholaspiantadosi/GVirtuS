#include <cuda_runtime.h>

void printArray(const int* dArray, int size) {
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

void printArrayF(const float* dArray, int size) {
    float hArray[size];
    cudaMemcpy(hArray, dArray, size * sizeof(float), cudaMemcpyDeviceToHost);
    printf("[");
    for (int i = 0; i < size; i++) {
        printf("%f", hArray[i]);
        if (i < (size - 1)) {
            printf(", ");
        }
    }
    printf("]\n");
}