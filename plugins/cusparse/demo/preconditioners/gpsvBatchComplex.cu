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

    cuComplex hds[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(8, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(7, 0)};
    cuComplex hdl[] = {make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(1, 0)};
    cuComplex hd[] = {make_cuComplex(1, 0), make_cuComplex(4, 0), make_cuComplex(6, 0), make_cuComplex(9, 0), make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(1, 0)};
    cuComplex hdu[] = {make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(7, 0), make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(3, 0), make_cuComplex(1, 0), make_cuComplex(0, 0)};
    cuComplex hdw[] = {make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex hx[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(3, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(4, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(6, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(7, 0), make_cuComplex(8, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(1, 0),
                      make_cuComplex(2, 0), make_cuComplex(1, 0), make_cuComplex(3, 0), make_cuComplex(4, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(5, 0), make_cuComplex(1, 0),
                      make_cuComplex(1, 0), make_cuComplex(1, 0), make_cuComplex(6, 0), make_cuComplex(7, 0)};

    cuComplex hx_result[] = {make_cuComplex(2.000000, 0), make_cuComplex(-0.436364, 0), make_cuComplex(1.000000, 0), make_cuComplex(0.381818, 0),
                            make_cuComplex(-1.000000, 0), make_cuComplex(3.363636, 0), make_cuComplex(2.000000, 0), make_cuComplex(-2.036364, 0),
                            make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(5.000000, 0), make_cuComplex(6.000000, 0),
                            make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(7.000000, 0), make_cuComplex(8.000000, 0),
                            make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0),
                            make_cuComplex(2.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(3.000000, 0), make_cuComplex(4.000000, 0),
                            make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(5.000000, 0), make_cuComplex(1.000000, 0),
                            make_cuComplex(1.000000, 0), make_cuComplex(1.000000, 0), make_cuComplex(6.000000, 0), make_cuComplex(7.000000, 0)};

    // Device memory management
    cuComplex *ddl, *dd, *ddu, *dx, *dds, *ddw;

    CHECK_CUDA( cudaMalloc((void**) &dds,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddw,  m * batchCount * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dx, m * n * batchCount * sizeof(cuComplex)) );

    CHECK_CUDA( cudaMemcpy(dds, hds, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddw, hdw, m * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * n * batchCount * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseCgpsvInterleavedBatch_bufferSizeExt(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseCgpsvInterleavedBatch(handle, algo, m, dds, ddl, dd, ddu, ddw, dx, batchCount, pBuffer);

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