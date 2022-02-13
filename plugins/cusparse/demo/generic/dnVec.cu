#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

int main(void) {
    int64_t size = 10;

    float hValues[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    float *dValues;

    CHECK_CUDA(cudaMalloc((void**) &dValues, size * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dValues, hValues, size * sizeof(float), cudaMemcpyHostToDevice) );

    cusparseDnVecDescr_t dnVecDescr;
    CHECK_CUSPARSE(cusparseCreateDnVec(&dnVecDescr, size, dValues, CUDA_R_32F));

    int64_t size_toverify;
    float hValues_toverify[size];
    float *dValues_toverify;
    cudaDataType valueType_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dValues_toverify, size * sizeof(float)));

    CHECK_CUSPARSE(cusparseDnVecGet(dnVecDescr, &size_toverify, (void**)&dValues_toverify, &valueType_toverify));

    CHECK_CUDA(cudaMemcpy(hValues_toverify, dValues_toverify, size * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (size_toverify != size) {
        correct = 0;
    }
    for (int i = 0; i < size; i++) {
        if((fabs(hValues_toverify[i] - hValues[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (valueType_toverify != CUDA_R_32F) {
        correct = 0;
    }

    float hValues2[] = {11, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float *dValues2;
    CHECK_CUDA(cudaMalloc((void**) &dValues2, size * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dValues2, hValues2, size * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseDnVecSetValues(dnVecDescr, dValues2));
    float hValues2_toverify[size];
    float *dValues2_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dValues2_toverify, size * sizeof(float)));
    CHECK_CUSPARSE(cusparseDnVecGetValues(dnVecDescr, (void**)&dValues2_toverify));
    CHECK_CUDA(cudaMemcpy(hValues2_toverify, dValues2_toverify, size * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < size; i++) {
        if((fabs(hValues2_toverify[i] - hValues2[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("dnVec test PASSED\n");
    else
        printf("dnVec test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroyDnVec(dnVecDescr));
    CHECK_CUDA(cudaFree(dValues) );

    return EXIT_SUCCESS;
}
