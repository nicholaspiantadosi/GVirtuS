#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverSp.h>         // cusolverSp
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <cusparse.h>

int main(void) {

    cusolverSpHandle_t handle = NULL;
    cudaStream_t streamIn = NULL;

    int correct = 1;

    cusolverStatus_t cs = cusolverSpCreate(&handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverSpSetStream(handle, streamIn);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    /*
    int m = 4;
    int nnzA = 9;
    cusparseMatDescr_t descrA = NULL;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    //float hCsrValA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int hCsrRowPtrA[] = {0, 3, 4, 7};
    const int hCsrEndPtrA[] = {2, 3, 6, 8};
    const int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};
    int issym;
    cs = cusolverSpXcsrissymHost(handle, m, nnzA, descrA, hCsrRowPtrA, hCsrEndPtrA, hCsrColIndA, &issym);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    correct = (issym == 0);
     */

    cs = cusolverSpDestroy(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_create_and_destroy test PASSED\n");
    } else {
        printf("cusolver_create_and_destroy test FAILED\n");
    }

    cudaStreamDestroy(streamIn);

    return EXIT_SUCCESS;
}