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
    const int nnz = 9;

    double hCsrValA[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    int hCsrRowPtrA[] = {1, 4, 5, 8, 10};
    int hCsrColIndA[] = {1, 3, 4, 2, 1, 3, 4, 2, 4};

    double hx[] = {1, 2, 3, 4};
    double hy[4];

    double hy_result[] = {1.526316, 0.500000, -0.105263, 0.000000};

    // Device memory management
    double *dCsrValA, *dx, *dy, *dz;
    int *dCsrRowPtrA, *dCsrColIndA;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dx,  m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dy,  m * sizeof(double)));
    CHECK_CUDA( cudaMalloc((void**) &dz,  m * sizeof(double)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * sizeof(double), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    //cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // Suppose that A is m x m sparse matrix represented by CSR format,
    // Assumption:
    // - handle is already created by cusparseCreate(),
    // - (d_csrRowPtr, d_csrColInd, d_csrVal) is CSR of A on device memory,
    // - d_x is right hand side vector on device memory,
    // - d_y is solution vector on device memory.
    // - d_z is intermediate result on device memory.

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    csric02Info_t info_M  = 0;
    csrsv2Info_t  info_L  = 0;
    csrsv2Info_t  info_Lt = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_Lt;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const double alpha = 1.;
    const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has non-unit diagonal
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csric02 and two info's for csrsv2
    cusparseCreateCsric02Info(&info_M);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_Lt);

    // step 3: query how much memory used in csric02 and csrsv2, and allocate the buffer
    cusparseDcsric02_bufferSize(handle, m, nnz,
                                descr_M, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M, &pBufferSize_M);
    cusparseDcsrsv2_bufferSize(handle, trans_L, m, nnz,
                               descr_L, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_L, &pBufferSize_L);
    cusparseDcsrsv2_bufferSize(handle, trans_Lt, m, nnz,
                               descr_L, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_Lt, &pBufferSize_Lt);

    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of csric02 and csrsv2 simultaneously.

    cusparseDcsric02_analysis(handle, m, nnz, descr_M,
                              dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M,
                              policy_M, pBuffer);
    cusparseStatus_t status = cusparseXcsric02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseDcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA,
                             info_L, policy_L, pBuffer);

    cusparseDcsrsv2_analysis(handle, trans_Lt, m, nnz, descr_L,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA,
                             info_Lt, policy_Lt, pBuffer);

    // step 5: M = L * L'
    cusparseDcsric02(handle, m, nnz, descr_M,
                     dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M, policy_M, pBuffer);

    status = cusparseXcsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseDcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, info_L,
                          dx, dz, policy_L, pBuffer);

    // step 7: solve L'*y = z
    cusparseDcsrsv2_solve(handle, trans_Lt, m, nnz, &alpha, descr_L,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, info_Lt,
                          dz, dy, policy_Lt, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hy, dy, m * sizeof(double), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if((fabs(hy[i] - hy_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csric02 test PASSED\n");
    else
        printf("csric02 test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_M));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
    CHECK_CUSPARSE(cusparseDestroyCsric02Info(info_M));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_L));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_Lt));
    CHECK_CUDA(cudaFree(dCsrValA) );
    CHECK_CUDA(cudaFree(dCsrRowPtrA) );
    CHECK_CUDA(cudaFree(dCsrColIndA) );
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dz));

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}