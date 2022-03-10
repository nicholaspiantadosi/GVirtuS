#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int lda = 3;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hA_result[] = {1, 0, 0, 0, 0, 0, 0, 0, 0};

    float *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));
    cusolverDnParams_t params = NULL;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    size_t workspaceInBytes;
    CUSOLVER_CHECK(cusolverDnPotrf_bufferSize(handle, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, dA, lda, CUDA_R_32F, &workspaceInBytes));

    void *pBuffer;
    cudaMalloc((void**)&pBuffer, workspaceInBytes);

    int *info;
    CUDA_CHECK( cudaMalloc((void**) &info, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnPotrf(handle, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, dA, lda, CUDA_R_32F, pBuffer, workspaceInBytes, info));
    int hInfo;
    CUDA_CHECK( cudaMemcpy(&hInfo, info, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*lda];
    CUDA_CHECK( cudaMemcpy(values, dA, sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hInfo == 0);
    for (int i = 0; i < n * lda; i++) {
        printf("%f == %f\n", values[i], hA_result[i]);
        if (fabsf(values[i] - hA_result[i]) > 0.05) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("dnpotrf test PASSED\n");
    } else {
        printf("dnpotrf test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}