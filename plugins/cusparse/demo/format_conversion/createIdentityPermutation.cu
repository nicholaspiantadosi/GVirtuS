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
    int n = 5;
    int hp[n];

    int hp_result[] = {0, 1, 2, 3, 4};

    // Device memory management
    int *dp;

    CHECK_CUDA( cudaMalloc((void**) &dp,  n * sizeof(int)));

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseCreateIdentityPermutation(handle, n, dp);

    // device result check
    CHECK_CUDA( cudaMemcpy(hp, dp, n * sizeof(int), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < n; i++) {
        if((fabs(hp[i] - hp_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("createIdentityPermutation test PASSED\n");
    else
        printf("createIdentityPermutation test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(dp) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}