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

    cuComplex hCsrValA[] = {
            make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0), make_cuComplex(5, 0),
            make_cuComplex(6, 0), make_cuComplex(7, 0), make_cuComplex(8, 0), make_cuComplex(9, 0)};

    int hCsrRowPtrA[] = {0, 3, 4, 7, 9};
    int hCsrColIndA[] = {0, 2, 3, 1, 0, 2, 3, 1, 3};

    cuComplex hB[] = {
            make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(6, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(8, 0)};

    cuComplex hC[] = {
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};
    cuComplex hC_result[] = {
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0),
            make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(0, 0)};

    cuComplex alpha = make_cuComplex(1, 0);

    // Device memory management
    cuComplex *dCsrValA;
    int *dCsrRowPtrA, *dCsrColIndA;
    cuComplex *dB, *dC;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dB,  m * nrhs * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dC,  m * nrhs * sizeof(cuComplex)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dB, hB, m * nrhs * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dC, hC, m * nrhs * sizeof(cuComplex), cudaMemcpyHostToDevice) );

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

    cusparseCcsrsm2_bufferSizeExt(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz,  &alpha, descrA,
                               dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, &pBufferSize);

    cudaMalloc((void**)&pBuffer, pBufferSize);

    cusparseCcsrsm2_analysis(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz, &alpha, descrA,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    cusparseStatus_t status = cusparseXcsrsm2_zeroPivot(handle, info, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    // WHERE IS C???
    cusparseCcsrsm2_solve(handle, algo, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, m, nrhs, nnz, &alpha, descrA,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, dB, ldb, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer);

    status = cusparseXcsrsm2_zeroPivot(handle, info, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is missing\n", numerical_zero, numerical_zero);
    }

    // device result check
    CHECK_CUDA( cudaMemcpy(hC, dC, m * nrhs * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m * nrhs; i++) {
        if((fabs(hC[i].x - hC_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csrm2 test PASSED\n");
    else
        printf("csrm2 test FAILED: wrong result\n");

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