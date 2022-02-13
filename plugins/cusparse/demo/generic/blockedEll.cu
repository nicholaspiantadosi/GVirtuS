#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(void) {
    int64_t rows = 9;
    int64_t cols = 9;
    int64_t ellBlockSize = 3;
    int64_t ellCols = 6;

    float hEllValue[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                         36, 37, 38, 42, 43, 44, 48, 49, 50, 0, 0, 0, 0, 0, 0};
    int hEllColInd[] = {0, 1, 0, 2, 2, -1};

    float *dEllValue;
    int *dEllColInd;

    CHECK_CUDA(cudaMalloc((void**) &dEllValue, rows * ellCols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dEllColInd, (ellCols / ellBlockSize) * (rows / ellBlockSize) * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dEllValue, hEllValue, rows * ellCols * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dEllColInd, hEllColInd, (ellCols / ellBlockSize) * (rows / ellBlockSize) * sizeof(int), cudaMemcpyHostToDevice) );

    cusparseSpMatDescr_t spMatDescr;
    CHECK_CUSPARSE(cusparseCreateBlockedEll(&spMatDescr, rows, cols, ellBlockSize, ellCols, dEllColInd, dEllValue, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int64_t rows_toverify;
    int64_t cols_toverify;
    cusparseIndexBase_t idxBase_toverify;

    int correct = 1;

    /*
    int64_t ellBlockSize_toverify;
    int64_t ellCols_toverify;
    float hEllValue_toverify[rows * ellCols];
    int hEllColInd_toverify[(ellCols / ellBlockSize) * (rows / ellBlockSize)];
    cusparseIndexType_t idxType_toverify;
    cudaDataType valueType_toverify;

    float *dEllValue_toverify;
    int *dEllColInd_toverify;

    CHECK_CUDA(cudaMalloc((void**) &dEllValue_toverify, rows * ellCols * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dEllColInd_toverify, (ellCols / ellBlockSize) * (rows / ellBlockSize) * sizeof(int)));

    CHECK_CUSPARSE(cusparseBlockedEllGet(spMatDescr, &rows_toverify, &cols_toverify, &ellBlockSize_toverify, &ellCols_toverify, (void**)&dEllColInd_toverify, (void**)&dEllValue_toverify, &idxType_toverify, &idxBase_toverify, &valueType_toverify));

    CHECK_CUDA(cudaMemcpy(hEllValue_toverify, dEllValue_toverify, rows * ellCols * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hEllColInd_toverify, dEllColInd_toverify, (ellCols / ellBlockSize) * (rows / ellBlockSize) * sizeof(int), cudaMemcpyDeviceToHost) );

    if (rows_toverify != rows) {
        correct = 0;
    }
    if (cols_toverify != cols) {
        correct = 0;
    }
    if (ellBlockSize_toverify != ellBlockSize) {
        correct = 0;
    }
    if (ellCols_toverify != ellCols) {
        correct = 0;
    }
    for (int i = 0; i < rows * ellCols; i++) {
        if((fabs(hEllValue_toverify[i] - hEllValue[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (ellCols / ellBlockSize) * (rows / ellBlockSize); i++) {
        if((fabs(hEllColInd_toverify[i] - hEllColInd[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (idxType_toverify != CUSPARSE_INDEX_32I) {
        correct = 0;
    }
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }
    if (valueType_toverify != CUDA_R_32F) {
        correct = 0;
    }
    */

    int64_t nnz_toverify;
    CHECK_CUSPARSE(cusparseSpMatGetSize(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify));
    if (rows_toverify != rows) {
        correct = 0;
    }
    if (cols_toverify != cols) {
        correct = 0;
    }
    if (nnz_toverify != 54) {
        correct = 0;
    }

    cusparseFormat_t format;
    CHECK_CUSPARSE(cusparseSpMatGetFormat(spMatDescr, &format));
    if (format != CUSPARSE_FORMAT_BLOCKED_ELL) {
        correct = 0;
    }

    CHECK_CUSPARSE(cusparseSpMatGetIndexBase(spMatDescr, &idxBase_toverify));
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }

    float hEllValue3[] = {51, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                         18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                         36, 37, 38, 42, 43, 44, 48, 49, 50, 0, 0, 0, 0, 0, 0};
    float *dEllValue3;
    float hEllValue3_toverify[rows * ellCols];
    float *dEllValue3_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dEllValue3, rows * ellCols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dEllValue3, hEllValue3, rows * ellCols * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseSpMatSetValues(spMatDescr, dEllValue3));
    CHECK_CUDA(cudaMalloc((void**) &dEllValue3_toverify, rows * ellCols * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpMatGetValues(spMatDescr, (void**)&dEllValue3_toverify));
    CHECK_CUDA(cudaMemcpy(hEllValue3_toverify, dEllValue3_toverify, rows * ellCols * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < 45; i++) {
        if((fabs(hEllValue3_toverify[i] - hEllValue3[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("blockedEll test PASSED\n");
    else
        printf("blockedEll test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroySpMat(spMatDescr));
    CHECK_CUDA(cudaFree(dEllValue) );
    CHECK_CUDA(cudaFree(dEllColInd) );
    CHECK_CUDA(cudaFree(dEllValue3) );

    return EXIT_SUCCESS;
}
