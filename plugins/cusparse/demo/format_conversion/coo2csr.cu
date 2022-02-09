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

    int hCooRowInd[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    const int nnz = 9;
    const int m = 4;

    int hCsrRowPtr[m + 1];

    int hCsrRowPtr_result[] = {0, 0, 1, 2, 3};

    // Device memory management

    int *dCooRowInd, *dCsrRowPtr;

    CHECK_CUDA( cudaMalloc((void**) &dCooRowInd, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtr, (m + 1) * sizeof(int)) );

    CHECK_CUDA( cudaMemcpy(dCooRowInd, hCooRowInd, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseStatus_t cs = cusparseXcoo2csr(handle, dCooRowInd, nnz, m, dCsrRowPtr, CUSPARSE_INDEX_BASE_ZERO);

    // device result check
    CHECK_CUDA( cudaMemcpy(hCsrRowPtr, dCsrRowPtr, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < (m + 1); i++) {
        if((fabs(hCsrRowPtr[i] - hCsrRowPtr_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("coo2csr test PASSED\n");
    else
        printf("coo2csr test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dCsrRowPtr));
    CHECK_CUDA(cudaFree(dCooRowInd));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}