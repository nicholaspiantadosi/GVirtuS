#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(void)
{
    int64_t size = 10;
    int64_t nnz = 5;
    int hIndices[] = {0, 2, 4, 6, 8};
    float hValues[] = {1, 2, 3, 4, 5};

    int *dIndices;
    float *dValues;

    CHECK_CUDA(cudaMalloc((void**) &dIndices, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dValues, nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dIndices, hIndices, nnz * sizeof(int), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dValues, hValues, nnz * sizeof(float), cudaMemcpyHostToDevice) );

    cusparseSpVecDescr_t spVecDescr;
    CHECK_CUSPARSE(cusparseCreateSpVec(&spVecDescr, size, nnz, dIndices, dValues, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    int64_t size_toverify;
    int64_t nnz_toverify;
    int hIndices_toverify[nnz];
    float hValues_toverify[nnz];
    cusparseIndexType_t idxType_toverify;
    cusparseIndexBase_t idxBase_toverify;
    cudaDataType valueType_toverify;

    int *dIndices_toverify;
    float *dValues_toverify;

    CHECK_CUDA(cudaMalloc((void**) &dIndices_toverify, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**) &dValues_toverify, nnz * sizeof(float)));

    CHECK_CUSPARSE(cusparseSpVecGet(spVecDescr, &size_toverify, &nnz_toverify, (void**)&dIndices_toverify, (void**)&dValues_toverify, &idxType_toverify, &idxBase_toverify, &valueType_toverify));

    CHECK_CUDA(cudaMemcpy(hIndices_toverify, dIndices_toverify, nnz * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA(cudaMemcpy(hValues_toverify, dValues_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (size_toverify != size) {
        correct = 0;
    }
    if (nnz_toverify != nnz) {
        correct = 0;
    }
    for (int i = 0; i < nnz; i++) {
        if((fabs(hIndices_toverify[i] - hIndices[i]) > 0.000001) || (fabs(hValues_toverify[i] - hValues[i]) > 0.000001)) {
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

    CHECK_CUSPARSE(cusparseSpVecGetIndexBase(spVecDescr, &idxBase_toverify));
    if (idxBase_toverify != CUSPARSE_INDEX_BASE_ZERO) {
        correct = 0;
    }

    float hValues2[] = {5, 4, 3, 2, 1};
    float *dValues2;
    CHECK_CUDA(cudaMalloc((void**) &dValues2, nnz * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dValues2, hValues2, nnz * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseSpVecSetValues(spVecDescr, dValues2));
    float *dValues2_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dValues2_toverify, nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseSpVecGetValues(spVecDescr, (void**) &dValues2_toverify));
    float hValues2_toverify[nnz];
    CHECK_CUDA(cudaMemcpy(hValues2_toverify, dValues2_toverify, nnz * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < nnz; i++) {
        if((fabs(hValues2_toverify[i] - hValues2[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("spvec test PASSED\n");
    else
        printf("spvec test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroySpVec(spVecDescr));
    CHECK_CUDA(cudaFree(dIndices) );
    CHECK_CUDA(cudaFree(dValues) );

    return EXIT_SUCCESS;
}
