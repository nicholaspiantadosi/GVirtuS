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

    cuDoubleComplex hds[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(7, 0)};
    cuDoubleComplex hdl[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(1, 0)};
    cuDoubleComplex hd[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(9, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0)};
    cuDoubleComplex hdu[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0)};
    cuDoubleComplex hdw[] = {make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex hx[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(4, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(6, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0),
                      make_cuDoubleComplex(2, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(1, 0),
                      make_cuDoubleComplex(1, 0), make_cuDoubleComplex(1, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(7, 0)};

    cuDoubleComplex hx_result[] = {make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(-0.436364, 0), make_cuDoubleComplex(-0.000000, 0), make_cuDoubleComplex(0.381818, 0),
                                    make_cuDoubleComplex(-0.000000, 0), make_cuDoubleComplex(3.363636, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(-2.036364, 0),
                                    make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(5.000000, 0), make_cuDoubleComplex(6.000000, 0),
                                    make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(7.000000, 0), make_cuDoubleComplex(8.000000, 0),
                                    make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0),
                                    make_cuDoubleComplex(2.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(3.000000, 0), make_cuDoubleComplex(4.000000, 0),
                                    make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(5.000000, 0), make_cuDoubleComplex(1.000000, 0),
                                    make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(6.000000, 0), make_cuDoubleComplex(7.000000, 0)};

    // Device memory management
    cuDoubleComplex *ddl, *dd, *ddu, *dx, *dds, *ddw;

    CHECK_CUDA( cudaMalloc((void**) &dds,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddw,  m * batchCount * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(cuDoubleComplex)) );

    CHECK_CUDA( cudaMemcpy(dds, hds, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddw, hdw, m * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseZgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseZgpsvInterleavedBatch(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, pBuffer);

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