#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverMg.h>         // cusolverMg
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../../cusolver_utils.h"

int main(void) {

    cusolverMgHandle_t handle = NULL;

    int correct = 1;

    cusolverStatus_t cs = cusolverMgCreate(&handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int deviceId[] = {0, 0, 0, 0};
    cs = cusolverMgDeviceSelect(handle, 4, deviceId);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cudaLibMgGrid_t grid = NULL;
    cs = cusolverMgCreateDeviceGrid(&grid, 1, 4, deviceId, CUDALIBMG_GRID_MAPPING_ROW_MAJOR);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cudaLibMgMatrixDesc_t desc;
    cs = cusolverMgCreateMatrixDesc(&desc, 1, 4, 1, 2, CUDA_R_32F, grid);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverMgDestroyMatrixDesc(desc);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverMgDestroyGrid(grid);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverMgDestroy(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("create_and_destroy test PASSED\n");
    } else {
        printf("create_and_destroy test FAILED\n");
    }

    return EXIT_SUCCESS;
}