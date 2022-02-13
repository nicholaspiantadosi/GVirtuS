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
    int64_t ld = cols;

    float hValues[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    float *dValues;

    CHECK_CUDA(cudaMalloc((void**) &dValues, rows * cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dValues, hValues, rows * cols * sizeof(float), cudaMemcpyHostToDevice) );

    cusparseDnMatDescr_t dnMatDescr;
    CHECK_CUSPARSE(cusparseCreateDnMat(&dnMatDescr, rows, cols, ld, dValues, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    int64_t rows_toverify;
    int64_t cols_toverify;
    int64_t ld_toverify;
    float hValues_toverify[rows * cols];
    float *dValues_toverify;
    cudaDataType valueType_toverify;
    cusparseOrder_t order_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dValues_toverify, rows * cols * sizeof(float)));

    CHECK_CUSPARSE(cusparseDnMatGet(dnMatDescr, &rows_toverify, &cols_toverify, &ld_toverify, (void**)&dValues_toverify, &valueType_toverify, &order_toverify));

    CHECK_CUDA(cudaMemcpy(hValues_toverify, dValues_toverify, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );

    int correct = 1;
    if (rows_toverify != rows) {
        correct = 0;
    }
    if (cols_toverify != cols) {
        correct = 0;
    }
    if (ld_toverify != ld) {
        correct = 0;
    }
    for (int i = 0; i < rows * cols; i++) {
        if((fabs(hValues_toverify[i] - hValues[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }
    if (valueType_toverify != CUDA_R_32F) {
        correct = 0;
    }
    if (order_toverify != CUSPARSE_ORDER_ROW) {
        correct = 0;
    }

    float hValues2[] = {11, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float *dValues2;
    CHECK_CUDA(cudaMalloc((void**) &dValues2, rows * cols * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dValues2, hValues2, rows * cols * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUSPARSE(cusparseDnMatSetValues(dnMatDescr, dValues2));
    float hValues2_toverify[rows * cols];
    float *dValues2_toverify;
    CHECK_CUDA(cudaMalloc((void**) &dValues2_toverify, rows * cols * sizeof(float)));
    CHECK_CUSPARSE(cusparseDnMatGetValues(dnMatDescr, (void**)&dValues2_toverify));
    CHECK_CUDA(cudaMemcpy(hValues2_toverify, dValues2_toverify, rows * cols * sizeof(float), cudaMemcpyDeviceToHost) );
    for (int i = 0; i < rows * cols; i++) {
        if((fabs(hValues2_toverify[i] - hValues2[i]) > 0.000001)) {
            correct = 0;
            break;
        }
    }

    if (correct)
        printf("dnMat test PASSED\n");
    else
        printf("dnMat test FAILED: wrong result\n");

    CHECK_CUSPARSE(cusparseDestroyDnMat(dnMatDescr));
    CHECK_CUDA(cudaFree(dValues) );
    CHECK_CUDA(cudaFree(dValues2) );

    return EXIT_SUCCESS;
}
