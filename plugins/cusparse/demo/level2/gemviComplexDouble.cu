#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{
    // Host problem definition - x = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0]
    cuDoubleComplex hX[] = { make_cuDoubleComplex(1,0), make_cuDoubleComplex(2,0), make_cuDoubleComplex(3,0), make_cuDoubleComplex(4,0) };
    int hXind[] = {0, 3, 4, 6};
    cuDoubleComplex hA[] = { make_cuDoubleComplex(1,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(2,0), make_cuDoubleComplex(3,0),
                       make_cuDoubleComplex(0,0), make_cuDoubleComplex(4,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0),
                       make_cuDoubleComplex(5,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(6,0), make_cuDoubleComplex(7,0),
                       make_cuDoubleComplex(0,0), make_cuDoubleComplex(8,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(9,0)};
    int m = 4;
    int n = 4;
    int nnz = 9;
    int lda = m;
    cuDoubleComplex alpha = make_cuDoubleComplex(1,0);
    cuDoubleComplex beta = make_cuDoubleComplex(0,0);
    cuDoubleComplex hY[] = { make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0), make_cuDoubleComplex(0,0) };
    cuDoubleComplex hY_result[] = { make_cuDoubleComplex(1,0), make_cuDoubleComplex(16,0), make_cuDoubleComplex(2,0), make_cuDoubleComplex(21,0) };

    // Device memory management
    cuDoubleComplex *dA;
    cuDoubleComplex *dX, *dY;
    int *dXind;

    CHECK_CUDA( cudaMalloc((void**) &dA,  m * n * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dX, n * sizeof(cuDoubleComplex)) );
    CHECK_CUDA( cudaMalloc((void**) &dXind, n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dY, m * sizeof(cuDoubleComplex)) );
    
    CHECK_CUDA( cudaMemcpy(dA, hA, m * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dX, hX, n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dXind, hXind, n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dY, hY, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    int pBufferSize = 0;
    void* dBuffer = NULL;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseZgemvi_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,m, n, nnz, &pBufferSize));
    CHECK_CUSPARSE(cusparseZgemvi(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &alpha, dA, lda, nnz, dX, dXind, &beta, dY, CUSPARSE_INDEX_BASE_ZERO, dBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (hY[i].x != hY_result[i].x) { // direct cuDoubleComplexing point comparison is not
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