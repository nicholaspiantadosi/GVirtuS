#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(void) {
    int64_t rows = 4;
    int64_t cols = 5;
    int64_t nnz = 9;

    float hCscValues[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCscColOffsetsA[] = {0, 2, 4, 6, 7, 9};
    int hCscRowIndA[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};

    float *dCscValues;
    int *dCscColOffsetsA, *dCscRowIndA;

    CHECK_CUDA(cudaMalloc((void**) &dCscValues, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCscColOffsetsA, (cols + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCscRowIndA, nnz * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dCscValues, hCscValues, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCscColOffsetsA, hCscColOffsetsA, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCscRowIndA, hCscRowIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    cusparseSpMatDescr_t spMatDescr;
    CHECK_CUSPARSE(cusparseCreateCsc(&spMatDescr, rows, cols, nnz, dCscColOffsetsA, dCscRowIndA, dCscValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    float hCscValues2[] = {10, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCscColOffsetsA2[] = {0, 2, 4, 6, 7, 9};
    int hCscRowIndA2[] = {0, 2, 0, 1, 1, 3, 2, 2, 3};
    float *dCscValues2;
    int *dCscColOffsetsA2, *dCscRowIndA2;
    CHECK_CUDA(cudaMalloc((void**) &dCscValues2, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCscColOffsetsA2, (cols + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCscRowIndA2, nnz * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dCscValues2, hCscValues2, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCscColOffsetsA2, hCscColOffsetsA2, (cols + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCscRowIndA2, hCscRowIndA2, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseCscSetPointers(spMatDescr, dCscColOffsetsA2, dCscRowIndA2, dCscValues2));

    int64_t rows_toverify;
    int64_t cols_toverify;
    int64_t nnz_toverify;
    CHECK_CUSPARSE(cusparseSpMatGetSize(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify));

    int correct = 1;
    if (rows_toverify != rows) {
        correct = 0;
    }
    if (cols_toverify != cols) {
        correct = 0;
    }
    if (nnz_toverify != nnz) {
        correct = 0;
    }

    cusparseFormat_t format;
    CHECK_CUSPARSE(cusparseSpMatGetFormat(spMatDescr, &format));
    if (format != CUSPARSE_FORMAT_CSC) {
        correct = 0;
    }

    cusparseIndexBase_t idxBase_toverify;
    CHECK_CUSPARSE(cusparseSpMatGetIndexBase(spMatDescr, &idxBase_toverify));
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }

    float hCscValues3[] = {11, 4, 2, 3, 5, 7, 8, 9, 6};
    float *dCscValues3;
    float hCscValues3_toverify[nnz];
    float *dCscValues3_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dCscValues3, nnz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dCscValues3, hCscValues3, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseSpMatSetValues(spMatDescr, dCscValues3));
    CHECK_CUDA(cudaMalloc((void**) &dCscValues3_toverify, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpMatGetValues(spMatDescr, (void**)&dCscValues3_toverify));
    CHECK_CUDA(cudaMemcpy(hCscValues3_toverify, dCscValues3_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCscValues3_toverify[i] - hCscValues3[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csc test PASSED\n");
    else
        printf("csc test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroySpMat(spMatDescr));
    CHECK_CUDA(cudaFree(dCscValues) );
    CHECK_CUDA(cudaFree(dCscRowIndA) );
    CHECK_CUDA(cudaFree(dCscColOffsetsA) );
    CHECK_CUDA(cudaFree(dCscRowIndA2) );
    CHECK_CUDA(cudaFree(dCscColOffsetsA2) );
    CHECK_CUDA(cudaFree(dCscValues2) );

    return EXIT_SUCCESS;
}
