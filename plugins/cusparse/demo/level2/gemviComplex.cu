#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

int main(int argn, char *argv[])
{
    // Host problem definition - x = [1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0]
    cuComplex hX[] = { make_cuComplex(1,0), make_cuComplex(2,0), make_cuComplex(3,0), make_cuComplex(4,0) };
    int hXind[] = {0, 3, 4, 6};
    cuComplex hA[] = { make_cuComplex(1,0), make_cuComplex(0,0), make_cuComplex(2,0), make_cuComplex(3,0),
                       make_cuComplex(0,0), make_cuComplex(4,0), make_cuComplex(0,0), make_cuComplex(0,0),
                       make_cuComplex(5,0), make_cuComplex(0,0), make_cuComplex(6,0), make_cuComplex(7,0),
                       make_cuComplex(0,0), make_cuComplex(8,0), make_cuComplex(0,0), make_cuComplex(9,0)};
    int m = 4;
    int n = 4;
    int nnz = 9;
    int lda = m;
    cuComplex alpha = make_cuComplex(1,0);
    cuComplex beta = make_cuComplex(0,0);
    cuComplex hY[] = { make_cuComplex(0,0), make_cuComplex(0,0), make_cuComplex(0,0), make_cuComplex(0,0) };
    cuComplex hY_result[] = { make_cuComplex(1,0), make_cuComplex(16,0), make_cuComplex(2,0), make_cuComplex(21,0) };

    // Device memory management
    cuComplex *dA;
    cuComplex *dX, *dY;
    int *dXind;

    CHECK_CUDA( cudaMalloc((void**) &dA,  m * n * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dX, n * sizeof(cuComplex)) );
    CHECK_CUDA( cudaMalloc((void**) &dXind, n * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dY, m * sizeof(cuComplex)) );
    
    CHECK_CUDA( cudaMemcpy(dA, hA, m * n * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dX, hX, n * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dXind, hXind, n * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dY, hY, m * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    int pBufferSize = 0;
    void* dBuffer = NULL;

    CHECK_CUSPARSE(cusparseCreate(&handle));
    CHECK_CUSPARSE(cusparseCgemvi_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,m, n, nnz, &pBufferSize));
    CHECK_CUSPARSE(cusparseCgemvi(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &alpha, dA, lda, nnz, dX, dXind, &beta, dY, CUSPARSE_INDEX_BASE_ZERO, dBuffer));

    // device result check
    CHECK_CUDA( cudaMemcpy(hY, dY, m * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if (hY[i].x != hY_result[i].x) { // direct cuComplexing point comparison is not
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