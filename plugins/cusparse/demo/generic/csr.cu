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

    float hCsrValues[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCsrRowOffsets[] = {0, 2, 4, 7, 9};
    int hCsrColInd[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};

    float *dCsrValues;
    int *dCsrRowOffsets, *dCsrColInd;

    CHECK_CUDA(cudaMalloc((void**) &dCsrValues, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrRowOffsets, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrColInd, nnz * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dCsrValues, hCsrValues, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrRowOffsets, hCsrRowOffsets, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrColInd, hCsrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    cusparseSpMatDescr_t spMatDescr;
    CHECK_CUSPARSE(cusparseCreateCsr(&spMatDescr, rows, cols, nnz, dCsrRowOffsets, dCsrColInd, dCsrValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int64_t rows_toverify;
    int64_t cols_toverify;
    int64_t nnz_toverify;
    float hCsrValues_toverify[nnz];
    int hCsrRowOffsets_toverify[nnz];
    int hCsrColInd_toverify[nnz];
    cusparseIndexType_t idxTypeRow_toverify;
    cusparseIndexType_t idxTypeCol_toverify;
    cusparseIndexBase_t idxBase_toverify;
    cudaDataType valueType_toverify;

    float *dCsrValues_toverify;
    int *dCsrRowOffsets_toverify, *dCsrColInd_toverify;

    CHECK_CUDA(cudaMalloc((void**) &dCsrValues_toverify, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrRowOffsets_toverify, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrColInd_toverify, nnz * sizeof(int)));

    CHECK_CUSPARSE(cusparseCsrGet(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify, (void**)&dCsrRowOffsets_toverify, (void**)&dCsrColInd_toverify, (void**)&dCsrValues_toverify, &idxTypeRow_toverify, &idxTypeCol_toverify, &idxBase_toverify, &valueType_toverify));

    CHECK_CUDA(cudaMemcpy(hCsrValues_toverify, dCsrValues_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCsrRowOffsets_toverify, dCsrRowOffsets_toverify, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCsrColInd_toverify, dCsrColInd_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

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
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrValues_toverify[i] - hCsrValues[i]) > 0.000001)
            || (fabs(hCsrColInd_toverify[i] - hCsrColInd[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (rows + 1); i++) {
        if((fabs(hCsrRowOffsets_toverify[i] - hCsrRowOffsets[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (idxTypeRow_toverify != CUSPARSE_INDEX_32I) {
        correct = 0;
    }
    if (idxTypeCol_toverify != CUSPARSE_INDEX_32I) {
        correct = 0;
    }
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }
    if (valueType_toverify != CUDA_R_32F) {
        correct = 0;
    }

    float hCsrValues2[] = {10, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCsrRowOffsets2[] = {0, 2, 4, 7, 9};
    int hCsrColInd2[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    float *dCsrValues2;
    int *dCsrRowOffsets2, *dCsrColInd2;
    float hCsrValues2_toverify[nnz];
    int hCsrRowOffsets2_toverify[nnz];
    int hCsrColInd2_toverify[nnz];
    float *dCsrValues2_toverify;
    int *dCsrRowOffsets2_toverify, *dCsrColInd2_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dCsrValues2, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrRowOffsets2, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrColInd2, nnz * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dCsrValues2, hCsrValues2, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrRowOffsets2, hCsrRowOffsets2, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCsrColInd2, hCsrColInd2, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseCsrSetPointers(spMatDescr, dCsrRowOffsets2, dCsrColInd2, dCsrValues2));
    CHECK_CUDA(cudaMalloc((void**) &dCsrValues2_toverify, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrRowOffsets2_toverify, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCsrColInd2_toverify, nnz * sizeof(int)));
    CHECK_CUSPARSE(cusparseCsrGet(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify, (void**)&dCsrRowOffsets2_toverify, (void**)&dCsrColInd2_toverify, (void**)&dCsrValues2_toverify, &idxTypeRow_toverify, &idxTypeCol_toverify, &idxBase_toverify, &valueType_toverify));
    CHECK_CUDA(cudaMemcpy(hCsrValues2_toverify, dCsrValues2_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCsrRowOffsets2_toverify, dCsrRowOffsets2_toverify, (rows + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCsrColInd2_toverify, dCsrColInd2_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrValues2_toverify[i] - hCsrValues2[i]) > 0.000001)
           || (fabs(hCsrColInd2_toverify[i] - hCsrColInd2[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < (rows + 1); i++) {
        if((fabs(hCsrRowOffsets2_toverify[i] - hCsrRowOffsets2[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    CHECK_CUSPARSE(cusparseSpMatGetSize(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify));
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
    if (format != CUSPARSE_FORMAT_CSR) {
        correct = 0;
    }

    CHECK_CUSPARSE(cusparseSpMatGetIndexBase(spMatDescr, &idxBase_toverify));
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }

    float hCsrValues3[] = {11, 4, 2, 3, 5, 7, 8, 9, 6};
    float *dCsrValues3;
    float hCsrValues3_toverify[nnz];
    float *dCsrValues3_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dCsrValues3, nnz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dCsrValues3, hCsrValues3, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseSpMatSetValues(spMatDescr, dCsrValues3));
    CHECK_CUDA(cudaMalloc((void**) &dCsrValues3_toverify, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpMatGetValues(spMatDescr, (void**)&dCsrValues3_toverify));
    CHECK_CUDA(cudaMemcpy(hCsrValues3_toverify, dCsrValues3_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCsrValues3_toverify[i] - hCsrValues3[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("csr test PASSED\n");
    else
        printf("csr test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroySpMat(spMatDescr));
    CHECK_CUDA(cudaFree(dCsrValues) );
    CHECK_CUDA(cudaFree(dCsrColInd) );
    CHECK_CUDA(cudaFree(dCsrRowOffsets) );
    CHECK_CUDA(cudaFree(dCsrColInd2) );
    CHECK_CUDA(cudaFree(dCsrRowOffsets2) );
    CHECK_CUDA(cudaFree(dCsrValues2) );

    return EXIT_SUCCESS;
}
