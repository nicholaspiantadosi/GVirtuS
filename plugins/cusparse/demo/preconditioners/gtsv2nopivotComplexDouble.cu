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

    cuDoubleComplex hdl[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};
    cuDoubleComplex hd[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(9, 0)};
    cuDoubleComplex hdu[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0)};

    int ldb = 4;
    cuDoubleComplex hB[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(3, 0),
                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(4, 0),
                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(6, 0),
                      make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0)};

    cuDoubleComplex hB_result[] = {make_cuDoubleComplex(1.000000, 0), make_cuDoubleComplex(0.500000, 0), make_cuDoubleComplex(-0.388889, 0), make_cuDoubleComplex(0.333333, 0),
                             make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(-0.518519, 0), make_cuDoubleComplex(0.444444, 0),
                             make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.055556, 0), make_cuDoubleComplex(0.666667, 0),
                             make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.000000, 0), make_cuDoubleComplex(0.129630, 0), make_cuDoubleComplex(0.888889, 0)};

    // Device memory management
    cuDoubleComplex *ddl, *dd, *ddu, *dB;

    CHECK_CUDA( cudaMalloc((void**) &ddl,  (m - 1) * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dd,  m * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &ddu,  (m - 1) * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dB, ldb * n * sizeof(cuDoubleComplex)) );

    CHECK_CUDA( cudaMemcpy(ddl, hdl, (m - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dd, hd, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(ddu, hdu, (m - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, ldb * n * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    size_t bufferSizeInBytes;
    void *pBuffer = 0;

    cusparseZgtsv2_nopivot_bufferSizeExt(handle, m, n, ddl, dd, ddu, dB, ldb, &bufferSizeInBytes);

    cudaMalloc((void**)&pBuffer, bufferSizeInBytes);

    cusparseZgtsv2_nopivot(handle, m, n, ddl, dd, ddu, dB, ldb, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hB, dB, ldb * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

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