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

    float hCooValA[] = {1, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCooRowIndA[] = {0, 0, 1, 1, 2, 2, 2, 3, 3};
    int hCooColIndA[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};

    float *dCooValA;
    int *dCooRowIndA, *dCooColIndA;

    CHECK_CUDA(cudaMalloc((void**) &dCooValA, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCooRowIndA, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCooColIndA, nnz * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(dCooValA, hCooValA, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCooRowIndA, hCooRowIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCooColIndA, hCooColIndA, nnz * sizeof(int), cudaMemcpyHostToDevice) );

    cusparseSpMatDescr_t spMatDescr;
    CHECK_CUSPARSE(cusparseCreateCoo(&spMatDescr, rows, cols, nnz, dCooRowIndA, dCooColIndA, dCooValA, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int64_t rows_toverify;
    int64_t cols_toverify;
    int64_t nnz_toverify;
    float hCooValA_toverify[nnz];
    int hCooRowIndA_toverify[nnz];
    int hCooColIndA_toverify[nnz];
    cusparseIndexType_t idxType_toverify;
    cusparseIndexBase_t idxBase_toverify;
    cudaDataType valueType_toverify;

    float *dCooValA_toverify;
    int *dCooRowIndA_toverify, *dCooColIndA_toverify;

    CHECK_CUDA(cudaMalloc((void**) &dCooValA_toverify, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCooRowIndA_toverify, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCooColIndA_toverify, nnz * sizeof(int)));

    CHECK_CUSPARSE(cusparseCooGet(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify, (void**)&dCooRowIndA_toverify, (void**)&dCooColIndA_toverify, (void**)&dCooValA_toverify, &idxType_toverify, &idxBase_toverify, &valueType_toverify));

    CHECK_CUDA(cudaMemcpy(hCooValA_toverify, dCooValA_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCooRowIndA_toverify, dCooRowIndA_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCooColIndA_toverify, dCooColIndA_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );

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
        if((fabs(hCooValA_toverify[i] - hCooValA[i]) > 0.000001)
            || (fabs(hCooRowIndA_toverify[i] - hCooRowIndA[i]) > 0.000001)
            || (fabs(hCooColIndA_toverify[i] - hCooColIndA[i]) > 0.000001)) {
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

    float hCooValA2[] = {10, 4, 2, 3, 5, 7, 8, 9, 6};
    int hCooRowIndA2[] = {0, 0, 1, 1, 2, 2, 2, 3, 3};
    int hCooColIndA2[] = {0, 1, 1, 2, 0, 3, 4, 2, 4};
    float *dCooValA2;
    int *dCooRowIndA2, *dCooColIndA2;
    float hCooValA2_toverify[nnz];
    int hCooRowIndA2_toverify[nnz];
    int hCooColIndA2_toverify[nnz];
    float *dCooValA2_toverify;
    int *dCooRowIndA2_toverify, *dCooColIndA2_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dCooValA2, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCooRowIndA2, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCooColIndA2, nnz * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(dCooValA2, hCooValA2, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCooRowIndA2, hCooRowIndA2, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dCooColIndA2, hCooColIndA2, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseCooSetPointers(spMatDescr, dCooRowIndA2, dCooColIndA2, dCooValA2));
    CHECK_CUDA(cudaMalloc((void**) &dCooValA2_toverify, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**) &dCooRowIndA2_toverify, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dCooColIndA2_toverify, nnz * sizeof(int)));
    CHECK_CUSPARSE(cusparseCooGet(spMatDescr, &rows_toverify, &cols_toverify, &nnz_toverify, (void**)&dCooRowIndA2_toverify, (void**)&dCooColIndA2_toverify, (void**)&dCooValA2_toverify, &idxType_toverify, &idxBase_toverify, &valueType_toverify));
    CHECK_CUDA(cudaMemcpy(hCooValA2_toverify, dCooValA2_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCooRowIndA2_toverify, dCooRowIndA2_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hCooColIndA2_toverify, dCooColIndA2_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCooValA2_toverify[i] - hCooValA2[i]) > 0.000001)
           || (fabs(hCooRowIndA2_toverify[i] - hCooRowIndA2[i]) > 0.000001)
           || (fabs(hCooColIndA2_toverify[i] - hCooColIndA2[i]) > 0.000001)) {
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
    if (format != CUSPARSE_FORMAT_COO) {
        correct = 0;
    }

    CHECK_CUSPARSE(cusparseSpMatGetIndexBase(spMatDescr, &idxBase_toverify));
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }

    float hCooValA3[] = {11, 4, 2, 3, 5, 7, 8, 9, 6};
    float *dCooValA3;
    float hCooValA3_toverify[nnz];
    float *dCooValA3_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dCooValA3, nnz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dCooValA3, hCooValA3, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseSpMatSetValues(spMatDescr, dCooValA3));
    CHECK_CUDA(cudaMalloc((void**) &dCooValA3_toverify, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpMatGetValues(spMatDescr, (void**)&dCooValA3_toverify));
    CHECK_CUDA(cudaMemcpy(hCooValA3_toverify, dCooValA3_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hCooValA3_toverify[i] - hCooValA3[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("coo test PASSED\n");
    else
        printf("coo test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroySpMat(spMatDescr));
    CHECK_CUDA(cudaFree(dCooValA) );
    CHECK_CUDA(cudaFree(dCooColIndA) );
    CHECK_CUDA(cudaFree(dCooRowIndA) );
    CHECK_CUDA(cudaFree(dCooColIndA2) );
    CHECK_CUDA(cudaFree(dCooRowIndA2) );
    CHECK_CUDA(cudaFree(dCooValA2) );

    return EXIT_SUCCESS;
}
