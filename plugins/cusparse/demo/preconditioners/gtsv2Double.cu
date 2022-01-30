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

    const int m = 4;
    const int n = 4;

    double hdl[] = {0, 0, 0, 0};
    double hd[] = {1, 4, 6, 9};
    double hdu[] = {0, 0, 7, 0};

    int ldb = 4;
    double hB[] = {1, 2, 0, 3,
                  0, 0, 0, 4,
                  0, 0, 5, 6,
                  0, 0, 7, 8};

    double hB_result[] = {1.000000, 0.500000, -0.388889, 0.333333,
                         0.000000, 0.000000, -0.518519, 0.444444,
                         0.000000, 0.000000, 0.055556, 0.666667,
                         0.000000, 0.000000, 0.129630, 0.888889};

    // Device memory management
    double *ddl, *dd, *ddu, *dB;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dB, ldb * n * sizeof(double)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, ldb * n * sizeof(double), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseDgtsv2_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseDgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hB, dB, ldb * n * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < (ldb * n); i++) {
        if((fabs(hB[i] - hB_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("gtsv2 test PASSED\n");
    else
        printf("gtsv2 test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(ddl));
    CHECK_CUDA(cudaFree(dd));
    CHECK_CUDA(cudaFree(ddu));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}