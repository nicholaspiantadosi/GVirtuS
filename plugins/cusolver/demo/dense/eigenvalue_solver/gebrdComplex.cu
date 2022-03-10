#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = 3;
    cuComplex hA[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(2, 0), make_cuComplex(5, 0), make_cuComplex(5, 0), make_cuComplex(3, 0), make_cuComplex(5, 0), make_cuComplex(12, 0)};;

    float hD_result[] = {-3.741657, -1.573688, -0.339662};
    float hE_result[] = {14.952305, 2.415928, 0.000000};
    cuComplex hTAUQ_result[] = {make_cuComplex(1.267261, 0), make_cuComplex(1.654486, 0), make_cuComplex(0.000000, 0)};
    cuComplex hTAUP_result[] = {make_cuComplex(1.482605, 0), make_cuComplex(0.000000, 0), make_cuComplex(0.000000, 0)};

    cuComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnCgebrd_bufferSize(handle, m, n, &Lwork));

    cuComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    float *dD, *dE;
    cuComplex *dTAUQ, *dTAUP;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUDA_CHECK( cudaMalloc((float**) &dD, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((float**) &dE, n * sizeof(float)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUQ, n * sizeof(cuComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUP, n * sizeof(cuComplex)));
    CUSOLVER_CHECK(cusolverDnCgebrd(handle, m, n, dA, lda, dD, dE, dTAUQ, dTAUP, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    float valuesD[n];
    float valuesE[n];
    cuComplex valuesTAUQ[n];
    cuComplex valuesTAUP[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dD, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, dE, n * sizeof(float), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUQ, dTAUQ, n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUP, dTAUP, n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    //int correct = (hdevInfo == 0);
    int correct = 1;
    for (int i = 0; i < n ; i++) {
        printf("%f \t %f \t %f \t %f\n", valuesD[i], valuesE[i], valuesTAUQ[i].x, valuesTAUP[i].x);
        if (fabsf(valuesD[i] - hD_result[i]) > 0.001
        || fabsf(valuesE[i] - hE_result[i]) > 0.001
        || fabsf(valuesTAUQ[i].x - hTAUQ_result[i].x) > 0.001
        || fabsf(valuesTAUP[i].x - hTAUP_result[i].x) > 0.001) {
            correct = 0;
            break;
        }
    }

    if (correct == 1) {
        printf("gebrd test PASSED\n");
    } else {
        printf("gebrd test FAILED\n");
    }

    CUSOLVER_CHECK(cusolverDnDestroy(handle));

    return EXIT_SUCCESS;

}