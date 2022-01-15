#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{
    // Host problem definition - x = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0]
    double hX[] = { 1, 2, 3, 4 };
    int hXind[] = {0, 3, 4, 6};
    double hA[] = { 1, 0, 2, 3,
                   0, 4, 0, 0,
                   5, 0, 6, 7,
                   0, 8, 0, 9 };
    int m = 4;
    int n = 4;
    int nnz = 9;
    int lda = m;
    double alpha = 1;
    double beta = 0;
    double hY[] = { 0, 0, 0, 0 };
    double hY_result[] = { 1, 16, 2, 21 };

    // Device memory management
    double *dA;
    double *dX, *dY;
    int *dXind;

    CHECK_CUDA( cudaMalloc((void**) &dA,  m * n * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dX, n * sizeof(double)) );
    CHECK_CUDA( cudaMalloc((void**) &dXind, n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dY, m * sizeof(double)) );

    CHECK_CUDA( cudaMemcpy(dA, hA, m * n * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dX, hX, n * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dXind, hXind, n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dY, hY, m * sizeof(double), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    int pBufferSize = 0;
    void* dBuffer = NULL;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseDgemvi_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,m, n, nnz, &pBufferSize));
    CHECK_CUSPARSE(cusparseDgemvi(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &alpha, dA, lda, nnz, dX, dXind, &beta, dY, CUSPARSE_INDEX_BASE_ZERO, dBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, m * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (hY[i] != hY_result[i]) { // direct doubleing point comparison is not
            correct = 0;             // reliable
            break;
        }
    }
    if (correct)
        printf("gemvi test PASSED\n");
    else
        printf("gemvi test FAILED: wrong result\n");

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) );
    CHECK_CUDA( cudaFree(dA) );
    CHECK_CUDA( cudaFree(dX) );
    CHECK_CUDA( cudaFree(dXind) );
    CHECK_CUDA( cudaFree(dY) );
    return EXIT_SUCCESS;

}