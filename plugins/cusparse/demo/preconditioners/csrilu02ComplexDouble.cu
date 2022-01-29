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

    cuDoubleComplex hCsrValA[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0), make_cuDoubleComplex(5, 0), make_cuDoubleComplex(6, 0), make_cuDoubleComplex(7, 0), make_cuDoubleComplex(8, 0), make_cuDoubleComplex(9, 0)};
    int hCsrRowPtrA[] = {1, 4, 5, 8, 10};
    int hCsrColIndA[] = {1, 3, 4, 2, 1, 3, 4, 2, 4};

    cuDoubleComplex hx[] = {make_cuDoubleComplex(1, 0), make_cuDoubleComplex(2, 0), make_cuDoubleComplex(3, 0), make_cuDoubleComplex(4, 0)};
    cuDoubleComplex hy[4];

    cuDoubleComplex hy_result[] = {make_cuDoubleComplex(0, 0), make_cuDoubleComplex(0.5, 0), make_cuDoubleComplex(0.5, 0), make_cuDoubleComplex(0, 0)};

    // Device memory management
    cuDoubleComplex *dCsrValA, *dx, *dy, *dz;
    int *dCsrRowPtrA, *dCsrColIndA;

    CHECK_CUDA( cudaMalloc((void**) &dCsrValA,  nnz * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dCsrRowPtrA, (m + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dCsrColIndA, nnz * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dx,  m * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dy,  m * sizeof(cuDoubleComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dz,  m * sizeof(cuDoubleComplex)));

    CHECK_CUDA( cudaMemcpy(dCsrValA, hCsrValA, nnz * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    //cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // Suppose that A is m x m sparse matrix represented by CSR format,
    // Assumption:
    // - handle is already created by cusparseCreate(),
    // - (dCsrRowPtrA, dCsrColIndA, dCsrValA) is CSR of A on device memory,
    // - dx is right hand side vector on device memory,
    // - dy is solution vector on device memory.
    // - dz is intermediate result on device memory.

    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    cusparseMatDescr_t descr_U = 0;
    csrilu02Info_t info_M  = 0;
    csrsv2Info_t  info_L  = 0;
    csrsv2Info_t  info_U  = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_U;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cuDoubleComplex alpha = make_cuDoubleComplex(1, 0);
    const cusparseSolvePolicy_t policy_M = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_U = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_U  = CUSPARSE_OPERATION_NON_TRANSPOSE;

    // step 1: create a descriptor which contains
    // - matrix M is base-1
    // - matrix L is base-1
    // - matrix L is lower triangular
    // - matrix L has unit diagonal
    // - matrix U is base-1
    // - matrix U is upper triangular
    // - matrix U has non-unit diagonal
    cusparseCreateMatDescr(&descr_M);
    cusparseSetMatIndexBase(descr_M, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_M, CUSPARSE_MATRIX_TYPE_GENERAL);

    cusparseCreateMatDescr(&descr_L);
    cusparseSetMatIndexBase(descr_L, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_L, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_L, CUSPARSE_FILL_MODE_LOWER);
    cusparseSetMatDiagType(descr_L, CUSPARSE_DIAG_TYPE_UNIT);

    cusparseCreateMatDescr(&descr_U);
    cusparseSetMatIndexBase(descr_U, CUSPARSE_INDEX_BASE_ONE);
    cusparseSetMatType(descr_U, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatFillMode(descr_U, CUSPARSE_FILL_MODE_UPPER);
    cusparseSetMatDiagType(descr_U, CUSPARSE_DIAG_TYPE_NON_UNIT);

    // step 2: create a empty info structure
    // we need one info for csrilu02 and two info's for csrsv2
    cusparseCreateCsrilu02Info(&info_M);
    cusparseCreateCsrsv2Info(&info_L);
    cusparseCreateCsrsv2Info(&info_U);

    // step 3: query how much memory used in csrilu02 and csrsv2, and allocate the buffer
    cusparseZcsrilu02_bufferSize(handle, m, nnz,
                                 descr_M, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M, &pBufferSize_M);
    cusparseZcsrsv2_bufferSize(handle, trans_L, m, nnz,
                               descr_L, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_L, &pBufferSize_L);
    cusparseZcsrsv2_bufferSize(handle, trans_U, m, nnz,
                               descr_U, dCsrValA, dCsrRowPtrA, dCsrColIndA, info_U, &pBufferSize_U);

    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_U));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on U
    // The lower(upper) triangular part of M has the same sparsity pattern as L(U),
    // we can do analysis of csrilu0 and csrsv2 simultaneously.

    cusparseZcsrilu02_analysis(handle, m, nnz, descr_M,
                               dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M,
                               policy_M, pBuffer);
    cusparseStatus_t status = cusparseXcsrilu02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseZcsrsv2_analysis(handle, trans_L, m, nnz, descr_L,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA,
                             info_L, policy_L, pBuffer);

    cusparseZcsrsv2_analysis(handle, trans_U, m, nnz, descr_U,
                             dCsrValA, dCsrRowPtrA, dCsrColIndA,
                             info_U, policy_U, pBuffer);

    // step 5: M = L * U
    cusparseZcsrilu02(handle, m, nnz, descr_M,
                      dCsrValA, dCsrRowPtrA, dCsrColIndA, info_M, policy_M, pBuffer);
    status = cusparseXcsrilu02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("U(%d,%d) is zero\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseZcsrsv2_solve(handle, trans_L, m, nnz, &alpha, descr_L,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, info_L,
                          dx, dz, policy_L, pBuffer);

    // step 7: solve U*y = z
    cusparseZcsrsv2_solve(handle, trans_U, m, nnz, &alpha, descr_U,
                          dCsrValA, dCsrRowPtrA, dCsrColIndA, info_U,
                          dz, dy, policy_U, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hy, dy, m * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if((fabs(hy[i].x - hy_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csrilu02 test PASSED\n");
    else
        printf("csrilu02 test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_M));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
    CHECK_CUSPARSE(cusparseDestroyCsrilu02Info(info_M));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_L));
    CHECK_CUSPARSE(cusparseDestroyCsrsv2Info(info_U));
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