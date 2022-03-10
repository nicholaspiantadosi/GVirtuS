#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include "../../cusolver_utils.h"
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    int m = 3;
    int n = 3;
    int lda = 3;
    cuDoubleComplex hA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(12, 0)};;

    double hD_result[] = {-0.339662, 0.000000, -0.339662};
    double hE_result[] = {14.952305, 2.415928, 0.000000};
    cuDoubleComplex hTAUQ_result[] = {make_cuDoubleComplex(1.267261, 0), make_cuDoubleComplex(1.654486, 0), make_cuDoubleComplex(0.000000, 0)};
    cuDoubleComplex hTAUP_result[] = {make_cuDoubleComplex(1.482605, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.000000, 0)};

    cuDoubleComplex *dA;
    CUDA_CHECK( cudaMalloc((void**) &dA, n * lda * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMemcpy(dA, hA, n * lda * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    cusolverDnHandle_t handle = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    int Lwork;
    CUSOLVER_CHECK(cusolverDnZgebrd_bufferSize(handle, m, n, &Lwork));

    cuDoubleComplex *Workspace;
    cudaMalloc((void**)&Workspace, Lwork);

    int *devInfo;
    double *dD, *dE;
    cuDoubleComplex *dTAUQ, *dTAUP;
    CUDA_CHECK( cudaMalloc((void**) &devInfo, sizeof(int)));
    CUDA_CHECK( cudaMalloc((double**) &dD, n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((double**) &dE, n * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUQ, n * sizeof(cuDoubleComplex)));
    CUDA_CHECK( cudaMalloc((void**) &dTAUP, n * sizeof(cuDoubleComplex)));
    CUSOLVER_CHECK(cusolverDnZgebrd(handle, m, n, dA, lda, dD, dE, dTAUQ, dTAUP, Workspace, Lwork, devInfo));
    int hdevInfo;
    CUDA_CHECK( cudaMemcpy(&hdevInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost) );
    double valuesD[n];
    double valuesE[n];
    cuDoubleComplex valuesTAUQ[n];
    cuDoubleComplex valuesTAUP[n];
    CUDA_CHECK( cudaMemcpy(valuesD, dD, n * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesE, dE, n * sizeof(double), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUQ, dTAUQ, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(valuesTAUP, dTAUP, n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

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