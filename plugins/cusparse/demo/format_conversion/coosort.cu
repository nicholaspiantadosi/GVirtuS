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

    /*
    float hA [] = {1, 0, 5, 0,
                  4, 2, 0, 0,
                  0, 3, 0, 9,
                  0, 0, 7, 0,
                  0, 0, 8, 6};
    */

    float hCooValA[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCooRowIndA[] = {0, 0, 1, 1, 2, 2, 2, 3, 3};
    int hCooColIndA[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};

    int hp[nnz];
    float hCooValASorted[nnz];

    int hCooRowIndA_resultByRow[] = {0, 0, 1, 1, 2, 2, 2, 3, 3};
    int hCooColIndA_resultByRow[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    int hp_resultByRow[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    float hCooValASorted_resultByRow[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};

    int hCooRowIndA_resultByColumn[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    int hCooColIndA_resultByColumn[] = {0, 0, 1, 1, 2, 2, 3, 4, 4};
    int hp_resultByColumn[] = {0, 4, 1, 2, 3, 7, 5, 6, 8};
    float hCooValASorted_resultByColumn[] = {1, 5, 4, 2, 3, 9, 7, 8, 6};

    // Device memory management
    float *dCooValA, *dCooValASorted;
    int *dCooRowIndA, *dCooColIndA, *dp;

    //CHECK_CUDA( cudaMalloc((void**) &dCooValA,  nnz * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dCooRowIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCooColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCooValA, nnz * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dCooValASorted, nnz * sizeof(float)) );
    CHECK_CUDA( cudaMalloc((void**) &dp, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCooRowIndA, hCooRowIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCooColIndA, hCooColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCooValA, hCooValA, nnz * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t pBufferSizeInBytes;
    void *pBuffer = 0;

    CHECK_CUSPARSE(cusparseXcoosort_bufferSizeExt(handle, m, n, nnz, dCooRowIndA, dCooColIndA, &pBufferSizeInBytes));

    pBufferSizeInBytes = 1152;

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSizeInBytes));

    CHECK_CUSPARSE(cusparseCreateIdentityPermutation(handle, nnz, dp));

    CHECK_CUSPARSE(cusparseXcoosortByRow(handle, m, n, nnz, dCooRowIndA, dCooColIndA, dp, pBuffer));

    CHECK_CUSPARSE(cusparseSgthr(handle, nnz, dCooValA, dCooValASorted, dp, CUSPARSE_INDEX_BASE_ZERO));

    // device result check
    CHECK_CUDA( cudaMemcpy(hCooRowIndA, dCooRowIndA, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCooColIndA, dCooColIndA, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hp, dp, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCooValASorted, dCooValASorted, nnz * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCooRowIndA[i] - hCooRowIndA_resultByRow[i]) > 0.000001)
        || (fabs(hCooColIndA[i] - hCooColIndA_resultByRow[i]) > 0.000001)
        || (fabs(hp[i] - hp_resultByRow[i]) > 0.000001)
        || (fabs(hCooValASorted[i] - hCooValASorted_resultByRow[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    CHECK_CUSPARSE(cusparseXcoosortByColumn(handle, m, n, nnz, dCooRowIndA, dCooColIndA, dp, pBuffer));

    CHECK_CUSPARSE(cusparseSgthr(handle, nnz, dCooValA, dCooValASorted, dp, CUSPARSE_INDEX_BASE_ZERO));

    // device result check
    CHECK_CUDA( cudaMemcpy(hCooRowIndA, dCooRowIndA, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCooColIndA, dCooColIndA, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hp, dp, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCooValASorted, dCooValASorted, nnz * sizeof(float), cudaMemcpyDeviceToHost) );

    for (int i = 0; i < nnz; i++) {
        if((fabs(hCooRowIndA[i] - hCooRowIndA_resultByColumn[i]) > 0.000001)
           || (fabs(hCooColIndA[i] - hCooColIndA_resultByColumn[i]) > 0.000001)
           || (fabs(hp[i] - hp_resultByColumn[i]) > 0.000001)
           || (fabs(hCooValASorted[i] - hCooValASorted_resultByColumn[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("coosort test PASSED\n");
    else
        printf("coosort test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCooRowIndA) );
    CHECK_CUDA(cudaFree(dCooColIndA) );
    CHECK_CUDA(cudaFree(dp) );
    CHECK_CUDA(cudaFree(dCooValA) );
    CHECK_CUDA(cudaFree(dCooValASorted) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}