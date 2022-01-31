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

    const int m = 3;
    const int nnz = 7;
    double fractionToColor = 0.8;

    double hCsrValA[] = {1, 1, -1, 1, 2, -1, 5};
    int hCsrRowPtrA[] = {0, 3, 5, 7};
    int hCsrColIndA[] = {0, 1, 2, 0, 1, 0, 2};

    int hColoring[m];
    int hReordering[] = {0, 0, 0};

    int hColoring_result[] = {3, 2, 4};
    int hReordering_result[] = {1, 0, 2};

    // Device memory management

    double *dCsrValA;
    int *dCsrRowPtrA, *dCsrColIndA;
    int ncolors;
    int *dColoring, *dReordering;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dColoring, m * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dReordering, m * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dReordering, hReordering, m * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseColorInfo_t info = 0;
    cusparseCreateColorInfo(&info);

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);
    //cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    //cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC);

    cusparseDcsrcolor(handle, m, nnz, descrA, dCsrValA, dCsrRowPtrA, dCsrColIndA, &fractionToColor, &ncolors, dColoring, dReordering, info);

    // device result check
    CHECK_CUDA( cudaMemcpy(hColoring, dColoring, m * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hReordering, dReordering, m * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if(hColoring[i] != hColoring_result[i]) {
            correct = 0;
            break;
        }
        if(hReordering[i] != hReordering_result[i]) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csrcolor test PASSED\n");
    else
        printf("csrcolor test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrA));
    CHECK_CUSPARSE(cusparseDestroyColorInfo(info));
    CHECK_CUDA(cudaFree(dCsrValA));
    CHECK_CUDA(cudaFree(dCsrRowPtrA));
    CHECK_CUDA(cudaFree(dCsrColIndA));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}