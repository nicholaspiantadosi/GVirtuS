#include <stdio.h>
#include <sys/time.h>
#include <cusparse.h>
#include <cuda_runtime_api.h>
#include "utilities.h"

int main(){
    int m = 6, n = 5;
    cusparseHandle_t  handle;
    CHECK_CUSPARSE( cusparseCreate(&handle) );
    cusparseMatDescr_t descrX;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrX));
    int total_nnz = 13;

    double *csrValX;
    int *csrRowPtrX;
    int *csrColIndX;
    CHECK_CUDA( cudaMalloc((void**) &csrValX, sizeof(double) * total_nnz) );
    CHECK_CUDA( cudaMalloc((void**) &csrRowPtrX, sizeof(int) * (m+1))) ;
    CHECK_CUDA( cudaMalloc((void**) &csrColIndX, sizeof(int) * total_nnz)) ;

    double hCsrVal[] = {1, 3, -4, 5, 2, 7, 8, 6, 9, 3.5, 5.5, 6.5, -9.9};
    int hCsrRowPtrX[] = {0, 2, 4, 7, 9, 11, 13};
    int hCsrColIndX[] = {0, 1, 1, 2, 0, 3, 4, 2, 4, 3, 4, 0, 2};

    CHECK_CUDA( cudaMemcpy(csrValX, hCsrVal, total_nnz * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csrRowPtrX, hCsrRowPtrX, (m + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA( cudaMemcpy(csrColIndX, hCsrColIndX, total_nnz * sizeof(int), cudaMemcpyHostToDevice) );

    double tol = 3.5;
    int *nnzPerRowY;
    int testNNZTotal;
    CHECK_CUDA( cudaMalloc((void**) &nnzPerRowY,  m * sizeof(int)));
    CHECK_CUSPARSE( cusparseDnnz_compress(handle, m, descrX, csrValX,
                                         csrRowPtrX, nnzPerRowY,
                                         &testNNZTotal, tol));
    double *csrValY;
    int *csrRowPtrY;
    int *csrColIndY;
    CHECK_CUDA( cudaMalloc((void**) &csrValY, sizeof(double) * testNNZTotal));
    CHECK_CUDA( cudaMalloc((void**) &csrRowPtrY, sizeof(int) * (m+1)));
    CHECK_CUDA( cudaMalloc((void**) &csrColIndY, sizeof(int) * testNNZTotal));

    CHECK_CUSPARSE( cusparseDcsr2csr_compress( handle, m, n, descrX, csrValX,
                                              csrColIndX, csrRowPtrX,
                                              total_nnz,  nnzPerRowY,
                                              csrValY, csrColIndY,
                                              csrRowPtrY, tol));

    int hNnzPerRowY[m];
    double hCsrValY[testNNZTotal];
    int hCsrRowPtrY[m + 1];
    int hCsrColIndY[testNNZTotal];

    CHECK_CUDA( cudaMemcpy(hNnzPerRowY, nnzPerRowY, m * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrValY, csrValY, testNNZTotal * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrY, csrRowPtrY, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndY, csrColIndY, testNNZTotal * sizeof(int), cudaMemcpyDeviceToHost) );

    int hNnzPerRowY_result[] = {0, 2, 2, 2, 1, 2};
    int hCsrRowPtrY_result[] = {0, 0, 2, 4, 6, 7, 9};

    int correct = 1;
    for (int i = 0; i < m; i++) {
        if((fabs(hNnzPerRowY[i] - hNnzPerRowY_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (m + 1); i++) {
        if((fabs(hCsrRowPtrY[i] - hCsrRowPtrY_result[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (correct)
        printf("csr2csr_compressed test PASSED\n");
    else
        printf("csr2csr_compressed test FAILED: wrong result\n");

    cusparseDestroy(handle);
    cudaFree(csrValX);
    cudaFree(csrRowPtrX);
    cudaFree(csrColIndX);
    cudaFree(csrValY);
    cudaFree(nnzPerRowY);
    cudaFree(csrRowPtrY);
    cudaFree(csrColIndY);
    return 0;
}
