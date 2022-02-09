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

    int hCsrRowPtr[] = {0, 0, 1, 2, 3};
    const int nnz = 9;
    const int m = 4;

    int hCooRowInd[nnz];

    int hCooRowInd_result[] = {1, 2, 3, 0, 0, 0, 0, 0, 0};

    // Device memory management

    int *dCsrRowPtr, *dCooRowInd;

    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCooRowInd, nnz * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCsrRowPtr, hCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseStatus_t cs = cusparseXcsr2coo(handle, dCsrRowPtr, nnz, m, dCooRowInd, CUSPARSE_INDEX_BASE_ZERO);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCooRowInd, dCooRowInd, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCooRowInd[i] - hCooRowInd_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csr2coo test PASSED\n");
    else
        printf("csr2coo test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCooRowInd));
    CHECK_CUDA(cudaFree(dCsrRowPtr));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}