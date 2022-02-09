#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(int argn, char *argv[])
{

    // Host problem definition
    int m = 4;
    int n = 5;
    int nnz = 9;
    int lda = m;

    cuDoubleComplex hCscValA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(9, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(6, 0)};
    int hCscRowIndA[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    int hCscColPtrA[] = {0, 2, 4, 6, 7, 9};

    cuDoubleComplex hA[m * n];

    cuDoubleComplex hA_result[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(0, 0),
                             make_cuDoubleComplex(4, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
                             make_cuDoubleComplex(0, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(9, 0),
                             make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(0, 0),
                             make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(6, 0)};

    // Device memory management
    cuDoubleComplex *dCscValA, *dA;
    int *dCscRowIndA, *dCscColPtrA;

    CHECK_CUDA( cudaMalloc((void**) &dCscValA,  nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCscRowIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtrA, (n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * n * sizeof(cuDoubleComplex)));

    CHECK_CUDA( cudaMemcpy(dCscValA, hCscValA, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscRowIndA, hCscRowIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscColPtrA, hCscColPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseZcsc2dense(handle, m, n, descrA, dCscValA, dCscRowIndA, dCscColPtrA, dA, lda);

    // device result check
    CHECK_CUDA( cudaMemcpy(hA, dA, lda * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < lda * n; i++) {
        if((fabs(hA[i].x - hA_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csc2dense test PASSED\n");
    else
        printf("csc2dense test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUDA(cudaFree(dCscValA) );
    CHECK_CUDA(cudaFree(dCscRowIndA) );
    CHECK_CUDA(cudaFree(dCscColPtrA) );
    CHECK_CUDA(cudaFree(dA) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}