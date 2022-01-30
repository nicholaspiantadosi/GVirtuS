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

    cuComplex hdl[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};
    cuComplex hd[] = {make_cuComplex(1, 0), make_cuComplex(4, 0), make_cuComplex(6, 0), make_cuComplex(9, 0)};
    cuComplex hdu[] = {make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(0, 0)};

    int ldb = 4;
    cuComplex hB[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(6, 0),
                      make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(8, 0)};

    cuComplex hB_result[] = {make_cuComplex(1.000000, 0), make_cuComplex(0.500000, 0), make_cuComplex(-0.388889, 0), make_cuComplex(0.333333, 0),
                             make_cuComplex(0.000000, 0), make_cuComplex(0.000000, 0), make_cuComplex(-0.518519, 0), make_cuComplex(0.444444, 0),
                             make_cuComplex(0.000000, 0), make_cuComplex(0.000000, 0), make_cuComplex(0.055556, 0), make_cuComplex(0.666667, 0),
                             make_cuComplex(0.000000, 0), make_cuComplex(0.000000, 0), make_cuComplex(0.129630, 0), make_cuComplex(0.888889, 0)};

    // Device memory management
    cuComplex *ddl, *dd, *ddu, *dB;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dB, ldb * n * sizeof(cuComplex)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, m * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, m * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, ldb * n * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseCgtsv2_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseCgtsv2(handle, m, n, ddl, dd, ddu, dB, ldb, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hB, dB, ldb * n * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < (ldb * n); i++) {
        if((fabs(hB[i].x - hB_result[i].x) > 0.000001)) {
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