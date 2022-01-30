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

    cuDoubleComplex hdl[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(1, 0)};
    cuDoubleComplex hd[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(9, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};
    cuDoubleComplex hdu[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};

    cuDoubleComplex hx[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(4, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(6, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0),
                      make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(1, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(7, 0)};

    cuDoubleComplex hx_result[] = {make_cuDoubleComplex(0.553030, 0), make_cuDoubleComplex(0.446970, 0), make_cuDoubleComplex(-0.340909, 0), make_cuDoubleComplex(0.371212, 0),
                             make_cuDoubleComplex(1.600000, 0), make_cuDoubleComplex(-0.600000, 0), make_cuDoubleComplex(0.200000, 0), make_cuDoubleComplex(3.800000, 0),
                             make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(5.000000, 0), make_cuDoubleComplex(6.000000, 0),
                             make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(7.000000, 0), make_cuDoubleComplex(8.000000, 0),
                             make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0),
                             make_cuDoubleComplex(2.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(3.000000, 0), make_cuDoubleComplex(4.000000, 0),
                             make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(5.000000, 0), make_cuDoubleComplex(1.000000, 0),
                             make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(6.000000, 0), make_cuDoubleComplex(7.000000, 0)};

    // Device memory management
    cuDoubleComplex *ddl, *dd, *ddu, *dx;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(cuDoubleComplex)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseZgtsv2StridedBatch_bufferSizeExt(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseZgtsv2StridedBatch(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hx, dx, m * n * batchCount * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < (m * n * batchCount); i++) {
        if((fabs(hx[i].x - hx_result[i].x) > 0.000001)) {
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