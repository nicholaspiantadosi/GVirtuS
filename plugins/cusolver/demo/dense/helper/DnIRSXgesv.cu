#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int n = 3;
    int nrhs = 3;
    int lda = n;
    int ldb = n;
    int ldx = n;
    float hA[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hB[] = {1, 2, 3, 2, 5, 5, 3, 5, 12};
    float hX[] = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    float hX_result[] = {1, 0, 0, 0, 1, 0, 0, 0, 1};

    float *dA, *dB, *dX;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dX, n * nrhs * sizeof(float)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n  * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(float), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, n * nrhs * sizeof(float), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    cusolverDnIRSParams_t gesv_irs_params = NULL;
    CUSOLVER_CHECK(cusolverDnIRSParamsCreate(&gesv_irs_params));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnIRSXgesv_bufferSize(handle, gesv_irs_params, n, nrhs, &lwork_bytes));
    printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    cusolverDnIRSInfos_t gesv_irs_infos = NULL;
    CUSOLVER_CHECK(cusolverDnIRSInfosCreate(&gesv_irs_infos));

    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnIRSXgesv(handle, gesv_irs_params, gesv_irs_infos, n, nrhs, dA, lda, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i], hX_result[i]);
        if (fabsf(values[i] - hX_result[i]) > 0.001) {
            correct = 0;
            //break;
        }
    }

    if (correct == 1) {
        printf("DnIRSXgesv test PASSED\n");
    } else {
        printf("DnIRSXgesv test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnIRSInfosDestroy(gesv_irs_infos));
    CUSOLVER_CHECK(cusolverDnIRSParamsDestroy(gesv_irs_params));
    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}