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

    cuComplex hdl[] = {make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(1, 0)};
    cuComplex hd[] = {make_cuComplex(1, 0), make_cuComplex(4, 0), make_cuComplex(6, 0), make_cuComplex(9, 0), make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(1, 0)};
    cuComplex hdu[] = {make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(7, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(3, 0), make_cuComplex(1, 0), make_cuComplex(1, 0)};

    cuComplex hx[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(3, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(4, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(6, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(7, 0), make_cuComplex(8, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0),
                      make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(3, 0), make_cuComplex(4, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(1, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(6, 0), make_cuComplex(7, 0)};

    cuComplex hx_result[] = {make_cuComplex(0.553030, 0), make_cuComplex(0.446970, 0), make_cuComplex(-0.340909, 0), make_cuComplex(0.371212, 0),
                             make_cuComplex(1.600000, 0), make_cuComplex(-0.600000, 0), make_cuComplex(0.200000, 0), make_cuComplex(3.800000, 0),
                             make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(5.000000, 0), make_cuComplex(6.000000, 0),
                             make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(7.000000, 0), make_cuComplex(8.000000, 0),
                             make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0),
                             make_cuComplex(2.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(3.000000, 0), make_cuComplex(4.000000, 0),
                             make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(5.000000, 0), make_cuComplex(1.000000, 0),
                             make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(6.000000, 0), make_cuComplex(7.000000, 0)};

    // Device memory management
    cuComplex *ddl, *dd, *ddu, *dx;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(cuComplex)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseCgtsv2StridedBatch_bufferSizeExt(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseCgtsv2StridedBatch(handle, m, ddl, dd, ddu, dx, batchCount, batchStride, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hx, dx, m * n * batchCount * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

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