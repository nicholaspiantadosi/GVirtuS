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

    const int blockSize = 2;
    const int mb = 2;
    const int nnzb = 4;

    cuComplex hBsrValA[] = {make_cuComplex(1, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(4, 0), make_cuComplex(2, 0), make_cuComplex(0, 0), make_cuComplex(3, 0), make_cuComplex(0, 0), make_cuComplex(5, 0), make_cuComplex(0, 0), make_cuComplex(0, 0), make_cuComplex(8, 0), make_cuComplex(6, 0), make_cuComplex(0, 0), make_cuComplex(7, 0), make_cuComplex(4, 0)};
    int hBsrRowPtrA[] = {1, 3, 5};
    int hBsrColIndA[] = {1, 2, 1, 2};

    const int m = mb * blockSize;

    cuComplex hx[] = {make_cuComplex(1, 0), make_cuComplex(2, 0), make_cuComplex(3, 0), make_cuComplex(4, 0)};
    cuComplex hy[4];

    cuComplex hy_result[] = {make_cuComplex(1.526316, 0), make_cuComplex(0.500000, 0), make_cuComplex(-0.105263, 0), make_cuComplex(0.000000, 0)};

    // Device memory management
    cuComplex *dBsrValA, *dx, *dy, *dz;
    int *dBsrRowPtrA, *dBsrColIndA;

    CHECK_CUDA( cudaMalloc((void**) &dBsrValA,  nnzb * (blockSize * blockSize) * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dBsrRowPtrA, (mb + 1) * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dBsrColIndA, nnzb * sizeof(int)) );
    CHECK_CUDA( cudaMalloc((void**) &dx,  m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dy,  m * sizeof(cuComplex)));
    CHECK_CUDA( cudaMalloc((void**) &dz,  m * sizeof(cuComplex)));

    CHECK_CUDA( cudaMemcpy(dBsrValA, hBsrValA, nnzb * (blockSize * blockSize) * sizeof(cuComplex), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrRowPtrA, hBsrRowPtrA, (mb + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dBsrColIndA, hBsrColIndA, nnzb * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(dx, hx, m * sizeof(cuComplex), cudaMemcpyHostToDevice) );

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    //cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);

    // Suppose that A is m x m sparse matrix represented by BSR format,
    // The number of block rows/columns is mb, and
    // the number of nonzero blocks is nnzb.
    // Assumption:
    // - handle is already created by cusparseCreate(),
    // - (dBsrRowPtrA, dBsrColIndA, dBsrValA) is BSR of A on device memory,
    // - dx is right hand side vector on device memory,
    // - dy is solution vector on device memory.
    // - dz is intermediate result on device memory.
    // - dx, dy and dz are of size m.
    cusparseMatDescr_t descr_M = 0;
    cusparseMatDescr_t descr_L = 0;
    bsric02Info_t info_M  = 0;
    bsrsv2Info_t  info_L  = 0;
    bsrsv2Info_t  info_Lt = 0;
    int pBufferSize_M;
    int pBufferSize_L;
    int pBufferSize_Lt;
    int pBufferSize;
    void *pBuffer = 0;
    int structural_zero;
    int numerical_zero;
    const cuComplex alpha = make_cuComplex(1, 0);
    const cusparseSolvePolicy_t policy_M  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_L  = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    const cusparseSolvePolicy_t policy_Lt = CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    const cusparseOperation_t trans_L  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseOperation_t trans_Lt = CUSPARSE_OPERATION_TRANSPOSE;
    const cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

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
    // we need one info for bsric02 and two info's for bsrsv2
    cusparseCreateBsric02Info(&info_M);
    cusparseCreateBsrsv2Info(&info_L);
    cusparseCreateBsrsv2Info(&info_Lt);

    // step 3: query how much memory used in bsric02 and bsrsv2, and allocate the buffer
    cusparseCbsric02_bufferSize(handle, dir, mb, nnzb,
                                descr_M, dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_M, &pBufferSize_M);
    cusparseCbsrsv2_bufferSize(handle, dir, trans_L, mb, nnzb,
                               descr_L, dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_L, &pBufferSize_L);
    cusparseCbsrsv2_bufferSize(handle, dir, trans_Lt, mb, nnzb,
                               descr_L, dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_Lt, &pBufferSize_Lt);

    pBufferSize = max(pBufferSize_M, max(pBufferSize_L, pBufferSize_Lt));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaMalloc((void**)&pBuffer, pBufferSize);

    // step 4: perform analysis of incomplete Cholesky on M
    //         perform analysis of triangular solve on L
    //         perform analysis of triangular solve on L'
    // The lower triangular part of M has the same sparsity pattern as L, so
    // we can do analysis of bsric02 and bsrsv2 simultaneously.

    cusparseCbsric02_analysis(handle, dir, mb, nnzb, descr_M,
                              dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_M,
                              policy_M, pBuffer);
    cusparseStatus_t status = cusparseXbsric02_zeroPivot(handle, info_M, &structural_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("A(%d,%d) is missing\n", structural_zero, structural_zero);
    }

    cusparseCbsrsv2_analysis(handle, dir, trans_L, mb, nnzb, descr_L,
                             dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize,
                             info_L, policy_L, pBuffer);

    cusparseCbsrsv2_analysis(handle, dir, trans_Lt, mb, nnzb, descr_L,
                             dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize,
                             info_Lt, policy_Lt, pBuffer);

    // step 5: M = L * L'
    cusparseCbsric02(handle, dir, mb, nnzb, descr_M,
                           dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_M, policy_M, pBuffer);
    status = cusparseXbsric02_zeroPivot(handle, info_M, &numerical_zero);
    if (CUSPARSE_STATUS_ZERO_PIVOT == status){
        printf("L(%d,%d) is not positive definite\n", numerical_zero, numerical_zero);
    }

    // step 6: solve L*z = x
    cusparseCbsrsv2_solve(handle, dir, trans_L, mb, nnzb, &alpha, descr_L,
                          dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_L,
                          dx, dz, policy_L, pBuffer);

    // step 7: solve L'*y = z
    cusparseCbsrsv2_solve(handle, dir, trans_Lt, mb, nnzb, &alpha, descr_L,
                          dBsrValA, dBsrRowPtrA, dBsrColIndA, blockSize, info_Lt,
                          dz, dy, policy_Lt, pBuffer);

    // device result check
    CHECK_CUDA( cudaMemcpy(hy, dy, m * sizeof(cuComplex), cudaMemcpyDeviceToHost) );

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if((fabs(hy[i].x - hy_result[i].x) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("bsric02 test PASSED\n");
    else
        printf("bsric02 test FAILED: wrong result\n");

    // step 6: free resources

    // device memory deallocation
    CHECK_CUDA(cudaFree(pBuffer));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_M));
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descr_L));
    CHECK_CUSPARSE(cusparseDestroyBsric02Info(info_M));
    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_L));
    CHECK_CUSPARSE(cusparseDestroyBsrsv2Info(info_Lt));
    CHECK_CUDA(cudaFree(dBsrValA) );
    CHECK_CUDA(cudaFree(dBsrRowPtrA) );
    CHECK_CUDA(cudaFree(dBsrColIndA) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    return EXIT_SUCCESS;
}