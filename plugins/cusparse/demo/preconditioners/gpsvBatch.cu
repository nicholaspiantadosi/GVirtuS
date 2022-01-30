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
    int algo = 0;

    float hds[] = {0, 0, 5, 8, 0, 0, 4, 7};
    float hdl[] = {0, 1, 1, 1, 0, 1, 5, 1};
    float hd[] = {1, 4, 6, 9, 1, 2, 1, 1};
    float hdu[] = {1, 1, 7, 0, 1, 3, 1, 0};
    float hdw[] = {2, 1, 0, 0, 1, 1, 0, 0};

    float hx[] = {1, 2, 1, 3,
                  1, 1, 1, 4,
                  1, 1, 5, 6,
                  1, 1, 7, 8,
                  1, 1, 1, 1,
                  2, 1, 3, 4,
                  1, 1, 5, 1,
                  1, 1, 6, 7};

    float hx_result[] = {2.000000, -0.436364, 1.000000, 0.381818,
                         -1.000000, 3.363636, 2.000000, -2.036364,
                         1.000000, 1.000000, 5.000000, 6.000000,
                         1.000000, 1.000000, 7.000000, 8.000000,
                         1.000000, 1.000000, 1.000000, 1.000000,
                         2.000000, 1.000000, 3.000000, 4.000000,
                         1.000000, 1.000000, 5.000000, 1.000000,
                         1.000000, 1.000000, 6.000000, 7.000000};

    // Device memory management
    float *ddl, *dd, *ddu, *dx, *dds, *ddw;

    CHECK_CUDA( cudaMalloc((void**) &dds,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &ddw,  m * batchCount * sizeof(float)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(float)) );

    CHECK_CUDA( cudaMemcpy(dds, hds, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddw, hdw, m * batchCount * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(float), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseSgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseSgpsvInterleavedBatch(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, pBuffer);

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
        printf("gpsvBatch test PASSED\n");
    else
        printf("gpsvBatch test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dds));
    CHECK_CUDA(cudaFree(ddl));
    CHECK_CUDA(cudaFree(dd));
    CHECK_CUDA(cudaFree(ddu));
    CHECK_CUDA(cudaFree(ddw));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}