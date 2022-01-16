#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "../level2/utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(int argn, char *argv[])
{

    // Host problem definition
    const int m = 4;
    const int nrhs = 4;
    const int nnz = 9;
    const int ldb = m; // leading dimension of B and C

    cuDoubleComplex hCsrValA[] = {
            make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(5, 0),
            make_cuDoubleComplex(6, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(9, 0)};
    
    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    cuDoubleComplex hB[] = {
            make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(3, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(4, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(6, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0)};

    cuDoubleComplex hC[] = {
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};
    cuDoubleComplex hC_result[] = {
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0),
            make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0, 0)};

    cuDoubleComplex alpha = make_cuDoubleComplex(1, 0);

    // Device memory management
    cuDoubleComplex *dCsrValA;
    int *dCsrRowPtrA, *dCsrColIndA;
    cuDoubleComplex *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dB,  m * nrhs * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dC,  m * nrhs * sizeof(cuDoubleComplex)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, m * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, m * nrhs * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrA = 0;
    cusparseCreateMatDescr(&descrA);

    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL );

    csrsm2Info_t info = 0;
    cusparseCreateCsrsm2Info(&info);
    size_t pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    int algo = 0;

    cusparseZcsrsm2_bufferSizeExt(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz,  &alpha, descrA,
                               dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, &pBufferSize);

    cudaMalloc((void**)&pBuffer, pBufferSize);

    cusparseZcsrsm2_analysis(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz, &alpha, descrA,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    cusparseStatus_t status = cusparseXcsrsm2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // WHERE IS C???
    cusparseZcsrsm2_solve(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz, &alpha, descrA,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    status = cusparseXcsrsm2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", numerical_zero, numerical_zero);
    }

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, m * nrhs * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m * nrhs; i++) {
        if((fabs(hC[i].x - hC_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("bsrmm test PASSED\n");
    else
        printf("bsrmm test FAILED: wrong result\n");

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    // device memory deallocation
    CHECK_CUDA( cudaFree(dCsrValA) );
    CHECK_CUDA( cudaFree(dCsrRowPtrA) );
    CHECK_CUDA( cudaFree(dCsrColIndA) );
    CHECK_CUDA( cudaFree(dB) );
    CHECK_CUDA( cudaFree(dC) );
    return EXIT_SUCCESS;
}