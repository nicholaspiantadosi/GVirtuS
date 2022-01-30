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
    int batchCount = 2;
    int batchStride = m;

    float hdl[] = {1, 1, 1, 1, 1, 1, 5, 1};
    float hd[] = {1, 4, 6, 9, 1, 2, 1, 1};
    float hdu[] = {1, 1, 7, 1, 1, 3, 1, 1};

    float hx[] = {1, 2, 1, 3,
                  1, 1, 1, 4,
                  1, 1, 5, 6,
                  1, 1, 7, 8,
                  1, 1, 1, 1,
                  2, 1, 3, 4,
                  1, 1, 5, 1,
                  1, 1, 6, 7};

    float hx_result[] = {0.553030, 0.446970, -0.340909, 0.371212,
                         1.600000, -0.600000, 0.200000, 3.800000,
                         1.000000, 1.000000, 5.000000, 6.000000,
                         1.000000, 1.000000, 7.000000, 8.000000,
                         1.000000, 1.000000, 1.000000, 1.000000,
                         2.000000, 1.000000, 3.000000, 4.000000,
                         1.000000, 1.000000, 5.000000, 1.000000,
                         1.000000, 1.000000, 6.000000, 7.000000};

    // Device memory management
    float *ddl, *dd, *ddu, *dx;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(float)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseSgtsv2StridedBatch_bufferSizeExt(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseSgtsv2StridedBatch(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hx, dx, m * n * batchCount * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < (m * n * batchCount); i++) {
        if((fabs(hx[i] - hx_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("gtsv2Batched test PASSED\n");
    else
        printf("gtsv2Batched test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(ddl));
    CHECK_CUDA(cudaFree(dd));
    CHECK_CUDA(cudaFree(ddu));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}