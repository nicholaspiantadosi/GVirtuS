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

    double hCscValA[] = {1, 5, 4, 2, 3, 9, 7, 8, 6};
    int hCscRowIndA[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    int hCscColPtrA[] = {0, 2, 4, 6, 7, 9};

    double hA[m * n];

    double hA_result[] = {1, 0, 5, 0,
                         4, 2, 0, 0,
                         0, 3, 0, 9,
                         0, 0, 7, 0,
                         0, 0, 8, 6};

    // Device memory management
    double *dCscValA, *dA;
    int *dCscRowIndA, *dCscColPtrA;

    CHECK_CUDA( cudaMalloc((void**) &dCscValA,  nnz * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCscRowIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCscColPtrA, (n + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dA,  lda * n * sizeof(double)));

    CHECK_CUDA( cudaMemcpy(dCscValA, hCscValA, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscRowIndA, hCscRowIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCscColPtrA, hCscColPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    cusparseDcsc2dense(handle, m, n, descrA, dCscValA, dCscRowIndA, dCscColPtrA, dA, lda);

    // device result check
    CHECK_CUDA( cudaMemcpy(hA, dA, lda * n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < lda * n; i++) {
        if((fabs(hA[i] - hA_result[i]) > 0.000001)) {
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