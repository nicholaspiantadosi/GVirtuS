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
    cuDoubleComplex hA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};
    cuDoubleComplex hB[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};
    cuDoubleComplex hX[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex hX_result[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0)};

    cuDoubleComplex *dA, *dB, *dX;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dX, n * nrhs * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n  * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnZEgesv_bufferSize(handle, n, nrhs, NULL, lda, NULL, NULL, ldb, NULL, ldx, NULL, &lwork_bytes));
    //lwork_bytes = 314880;
    printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    int *devIpiv;
    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnZEgesv(handle, n, nrhs, dA, lda, devIpiv, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuDoubleComplex values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i].x, hX_result[i].x);
        if (fabsf(values[i].x - hX_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("DnZEgesv test PASSED\n");
    } else {
        printf("DnZEgesv test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}