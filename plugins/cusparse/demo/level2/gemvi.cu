#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{
    // Host problem definition - x = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0]
    float hX[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    int hXind[] = {0, 3, 4, 6};
    float hA[] = { 1.0f, 0.0f, 2.0f, 3.0f,
                   0.0f, 4.0f, 0.0f, 0.0f,
                   5.0f, 0.0f, 6.0f, 7.0f,
                   0.0f, 8.0f, 0.0f, 9.0f };
    int m = 4;
    int n = 4;
    int nnz = 9;
    int lda = m;
    float alpha = 1.0f;
    float beta = 0.0f;
    float hY[] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float hY_result[] = { 1.0f, 16.0f, 2.0f, 21.0f };

    // Device memory management
    float *dA;
    float *dX, *dY;
    int *dXind;

    CHECK_CUDA( cudaMalloc((void**) &dA,  m * n * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dX, n * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dXind, n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dY, m * sizeof(float)) );

    CHECK_CUDA( cudaMemcpy(dA, hA, m * n * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dX, hX, n * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dXind, hXind, n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dY, hY, m * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    int pBufferSize = 0;
    void* dBuffer = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseSgemvi_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,m, n, nnz, &pBufferSize));

    CHECK_CUSPARSE(cusparseSgemvi(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &alpha, dA, lda, nnz, dX, dXind, &beta, dY, CUSPARSE_INDEX_BASE_ZERO, dBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, m * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (hY[i] != hY_result[i]) { // direct floating point comparison is not
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