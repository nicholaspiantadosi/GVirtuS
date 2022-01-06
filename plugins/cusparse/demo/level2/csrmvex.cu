#include <cuda_runtime.h>  // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>      // cusparseSpMV
#include <stdio.h>         // printf
#include <stdlib.h>        // EXIT_FAILURE
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}
//#ifndef USE_FLOAT
//#define DTYPE CUDA_R_16F
//typedef half dtype;
//#else
#define DTYPE CUDA_R_32F
typedef float dtype;
//#endif

int main() {
    // Host problem definition
    const int A_num_rows = 4;
    const int A_num_cols = 4;
    const int A_num_nnz  = 9;
    int   hA_csrOffsets[] = { 0, 3, 4, 7, 9 };
    int   hA_columns[]    = { 0, 2, 3, 1, 0, 2, 3, 1, 3 };
    dtype hA_values[]     = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                              6.0f, 7.0f, 8.0f, 9.0f };
    dtype hX[]            = { 1.0f, 2.0f, 3.0f, 4.0f };
    dtype yTemp[]         = { 19.0f, 0.0f, 0.0f, 0.0f };
    const dtype result[]  = { 19.0f, 8.0f, 51.0f, 52.0f };
    //--------------------------------------------------------------------------
    // Device memory management
    int   *dA_csrOffsets, *dA_columns;
    dtype *dA_values, *dX, *dY;
    CHECK_CUDA( cudaMalloc((void**) &dA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_num_nnz * sizeof(int)) )
    CHECK_CUDA( cudaMalloc((void**) &dA_values, A_num_nnz * sizeof(dtype)) )
    CHECK_CUDA( cudaMalloc((void**) &dX, A_num_cols * sizeof(dtype)) )
    CHECK_CUDA( cudaMalloc((void**) &dY, A_num_rows * sizeof(dtype)) )

    CHECK_CUDA( cudaMemcpy(dA_csrOffsets, hA_csrOffsets,
                           (A_num_rows + 1) * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_num_nnz * sizeof(int),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dA_values, hA_values,
                           A_num_nnz * sizeof(dtype), cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dX, hX, A_num_rows * sizeof(dtype),
                           cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dY, yTemp, A_num_rows * sizeof(dtype),
                           cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    // CUSPARSE APIs
    cusparseHandle_t     handle = 0;
    void*  dBuffer    = NULL;
    size_t bufferSize = 0;
    CHECK_CUSPARSE( cusparseCreate(&handle) )
#ifdef USE_SPMV
    float alpha = 1.0f;
    float beta  = 0.0f;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_num_nnz,
                                      dA_csrOffsets, dA_columns, dA_values,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, DTYPE) )
    // Create dense vector X
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecX, A_num_cols, dX, DTYPE) )
    // Create dense vector y
    CHECK_CUSPARSE( cusparseCreateDnVec(&vecY, A_num_rows, dY, DTYPE) )
    // allocate an external buffer if needed
    //printf("\nalpha: %f\n", alpha);
    CHECK_CUSPARSE( cusparseSpMV_bufferSize(
                                 handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, DTYPE,
                                 CUSPARSE_MV_ALG_DEFAULT, &bufferSize) )
    //printf("\nbufferSize: %d\n", bufferSize);
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )

    // execute SpMV
    CHECK_CUSPARSE( cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, vecX, &beta, vecY, DTYPE,
                                 CUSPARSE_MV_ALG_DEFAULT, dBuffer) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecX) )
    CHECK_CUSPARSE( cusparseDestroyDnVec(vecY) )
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
#else
    dtype alpha = 1.0f;
    dtype beta  = 0.0f;
    cusparseMatDescr_t descrA;
    CHECK_CUSPARSE( cusparseCreateMatDescr(&descrA) );
    CHECK_CUSPARSE( cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL) );
    CHECK_CUSPARSE( cusparseCsrmvEx_bufferSize(
            handle,
            CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
            A_num_rows, A_num_cols, A_num_nnz,
            &alpha, DTYPE,
            descrA, dA_values, DTYPE, dA_csrOffsets, dA_columns,
            dX, DTYPE,
            &beta, DTYPE,
            dY, DTYPE,
            DTYPE,
            &bufferSize) );
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );
    CHECK_CUSPARSE( cusparseCsrmvEx(
            handle,
            CUSPARSE_ALG_MERGE_PATH, CUSPARSE_OPERATION_NON_TRANSPOSE,
            A_num_rows, A_num_cols, A_num_nnz,
            &alpha, DTYPE,
            descrA, dA_values, DTYPE, dA_csrOffsets, dA_columns,
            dX, DTYPE,
            &beta, DTYPE,
            dY, DTYPE,
            DTYPE,
            dBuffer) );  // this is line 121
#endif
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroy(handle) )
    //--------------------------------------------------------------------------
    // device result check
    dtype hY[A_num_rows];
    CHECK_CUDA( cudaMemcpy(hY, dY, A_num_rows * sizeof(dtype),
                           cudaMemcpyDeviceToHost) )

    int correct = 1;
    for (int i = 0; i < A_num_rows; i++) {
        if (((float)hY[i]) != ((float)result[i])) {
            correct = 0;
            printf("hY[%d] = %f, result[%d] = %f\n", i, (float)hY[i], i, (float)result[i]);
            //break;
        }
    }
    if (correct)
        printf("spmv_example test PASSED\n");
    else
        printf("spmv_example test FAILED: wrong result\n");
    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dBuffer) )
    CHECK_CUDA( cudaFree(dA_csrOffsets) )
    CHECK_CUDA( cudaFree(dA_columns) )
    CHECK_CUDA( cudaFree(dA_values) )
    CHECK_CUDA( cudaFree(dX) )
    CHECK_CUDA( cudaFree(dY) )
    return EXIT_SUCCESS;
}