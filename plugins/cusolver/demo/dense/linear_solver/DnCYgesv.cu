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
    cuComplex hA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};
    cuComplex hB[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};
    cuComplex hX[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex hX_result[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(1, 0)};

    cuComplex *dA, *dB, *dX;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * n * sizeof(cuComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dB, n * nrhs * sizeof(cuComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dX, n * nrhs * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * n  * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dB, hB, n * nrhs * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dX, hX, n * nrhs * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    size_t lwork_bytes;
    CUSOLVER_CHECK(cusolverDnCYgesv_bufferSize(handle, n, nrhs, NULL, lda, NULL, NULL, ldb, NULL, ldx, NULL, &lwork_bytes));
    //lwork_bytes = 88832;
    printf("%d\n", lwork_bytes);

    void *dWorkspace;
    cudaMalloc((void**)&dWorkspace, lwork_bytes);

    int *devIpiv;
    int *devInfo;
    int niter;
    CUDA_CHECK( cudaMalloc((void**) &devIpiv, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUSOLVER_CHECK(cusolverDnCYgesv(handle, n, nrhs, dA, lda, devIpiv, dB, ldb, dX, ldx, dWorkspace, lwork_bytes, &niter, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    cuComplex values[n*nrhs];
    CUDA_CHECK( cudaMemcpy(values, dX, n * nrhs * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = (hdevInfo == 0);
    for (int i = 0; i < n * nrhs; i++) {
        printf("%f == %f\n", values[i].x, hX_result[i].x);
        if (fabsf(values[i].x - hX_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("DnCYgesv test PASSED\n");
    } else {
        printf("DnCYgesv test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}