#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    cusolverDnHandle_t handle = NULL;
    syevjInfo_t info = NULL;

    cusolverStatus_t cs = cusolverDnCreate(&handle);

    cs = cusolverDnCreateSyevjInfo(&info);

    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double tolerance = 0.1;
    cs = cusolverDnXsyevjSetTolerance(info, tolerance);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int max_sweeps = 99;
    cs = cusolverDnXsyevjSetMaxSweeps(info, max_sweeps);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int sort_eig = 1;
    cs = cusolverDnXsyevjSetSortEig(info, sort_eig);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double residual;
    cs = cusolverDnXsyevjGetResidual(handle, info, &residual);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int sweeps;
    cs = cusolverDnXsyevjGetSweeps(handle, info, &sweeps);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnDestroySyevjInfo(info);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_syevjinfo test PASSED\n");
    } else {
        printf("cusolver_syevjinfo test FAILED\n");
    }

    cusolverDnDestroy(handle);

    return EXIT_SUCCESS;
}