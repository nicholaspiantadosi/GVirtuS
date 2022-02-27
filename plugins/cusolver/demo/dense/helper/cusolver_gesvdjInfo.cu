#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    cusolverDnHandle_t handle = NULL;
    gesvdjInfo_t info = NULL;

    cusolverStatus_t cs = cusolverDnCreate(&handle);

    cs = cusolverDnCreateGesvdjInfo(&info);

    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double tolerance = 0.1;
    cs = cusolverDnXgesvdjSetTolerance(info, tolerance);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int max_sweeps = 99;
    cs = cusolverDnXgesvdjSetMaxSweeps(info, max_sweeps);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int sort_eig = 1;
    cs = cusolverDnXgesvdjSetSortEig(info, sort_eig);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double residual;
    cs = cusolverDnXgesvdjGetResidual(handle, info, &residual);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int sweeps;
    cs = cusolverDnXgesvdjGetSweeps(handle, info, &sweeps);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnDestroyGesvdjInfo(info);

    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_gesvdjInfo test PASSED\n");
    } else {
        printf("cusolver_gesvdjInfo test FAILED\n");
    }

    cusolverDnDestroy(handle);

    return EXIT_SUCCESS;
}