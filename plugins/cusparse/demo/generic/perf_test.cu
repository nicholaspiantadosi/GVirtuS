#include<stdio.h>
#include<stdlib.h>
#include<cusparse.h>
#include <time.h>

#include "utilities.h"
#include <cuda_runtime_api.h>

#include <limits>

void initializeMatrixZero(double *matrix, int M, int N)
{
    int i, j, k=0;
    for (i = 0; i < N; i++)
    {
        for (j = 0; j < M; j++)
        {
            matrix[k++]=0;
        }
    }
}

void initializeMatrixRandomSparse(double *matrix, int M, int N, int nnz)
{
	initializeMatrixZero(matrix, M, N);
	int i=0;
	double random_number;
	for (i = 0; i < nnz;) {
		int index = (int) (M * N * ((double) rand() / (RAND_MAX + 1.0)));
		if (matrix[index]) { 
			continue;
		}
		random_number = (double) rand() / ( (double) RAND_MAX / 100 ) + 1;
		matrix[index] = random_number;
		++i;
	}
}

void initializeMatrixRandom(double *matrix, int M, int N)
{
	int i,j,k=0;
	double random_number;
	for (i = 0; i < N; i++)
	{
		for (j = 0; j < M; j++)
		{
			random_number=(double)rand()/((double)RAND_MAX/(100)) + 1;
			matrix[k++]= random_number;
		}
	}
}

int main(int argn, char *argv[])
{
    // Host problem definition
    int m = 1000;
    int n = 1000;
    int nnz = 500;
    int lda = m;
    int ldb = m;
    int ldc = m;
    double alpha = 2.5f;
    double beta = 1.5f;

    double *hA = (double *)malloc(m*n*sizeof(double));
    initializeMatrixRandomSparse(hA, m, n, nnz);

    double *hB = (double *)malloc(m*n*sizeof(double));
    initializeMatrixRandom(hB, m, n);

    double *hC = (double *)malloc(m*n*sizeof(double));
    initializeMatrixRandom(hC, m, n);

    double threshold = 2;

    // Device memory management
    double *dA, *dB, *dC;
    double *dCsrValC;
    int *dCsrRowPtrC, *dCsrColIndC;

    clock_t tStart = clock();

    CHECK_CUDA(cudaMalloc((void**) &dA,  m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**)&dCsrRowPtrC, sizeof(int) * (m + 1)));

    CHECK_CUDA(cudaMalloc((void**) &dB,  m * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc((void**) &dC,  m * n * sizeof(double)));

    CHECK_CUDA(cudaMemcpy(dA, hA, m * n * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dB, hB, m * n * sizeof(double), cudaMemcpyHostToDevice) );
    CHECK_CUDA(cudaMemcpy(dC, hC, m * n * sizeof(double), cudaMemcpyHostToDevice) );

    clock_t tCopied2Device = clock();

    // CUSPARSE APIs
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseMatDescr_t descrC = 0;
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrC));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL ));

    size_t pBufferSize;
    void *pBuffer = 0;

    CHECK_CUSPARSE(cusparseDpruneDense2csr_bufferSizeExt(handle, m, n, dA, lda, &threshold, descrC, NULL, dCsrRowPtrC, NULL, &pBufferSize));

//    if(pBufferSize == 0) {
        pBufferSize = 512;
//    }

    CHECK_CUDA(cudaMalloc((void**)&pBuffer, pBufferSize));

    int nnzc;
    int *nnzTotalDevHostPtr = &nnzc;

    CHECK_CUSPARSE(cusparseDpruneDense2csrNnz(handle, m, n, dA, lda, &threshold, descrC, dCsrRowPtrC, nnzTotalDevHostPtr, pBuffer));

    nnzc = *nnzTotalDevHostPtr;

    CHECK_CUDA(cudaMalloc((void**)&dCsrValC, sizeof(double) * nnzc));
    CHECK_CUDA(cudaMalloc((void**)&dCsrColIndC, sizeof(int) * nnzc));

    CHECK_CUSPARSE(cusparseDpruneDense2csr(handle, m, n, dA, lda, &threshold, descrC, dCsrValC, dCsrRowPtrC, dCsrColIndC, pBuffer));

    clock_t tDense2Csr = clock();

    cusparseSpMatDescr_t matA;
    cusparseDnMatDescr_t matB, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;

    // Create sparse matrix A in CSR format
    CHECK_CUSPARSE( cusparseCreateCsr(&matA, m, n, nnz,
                                      dCsrRowPtrC, dCsrColIndC, dCsrValC,
                                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                      CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) );
    // Create dense matrix B
    CHECK_CUSPARSE( cusparseCreateDnMat(&matB, m, n, ldb, dB,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) );
    // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, m, n, ldc, dC,
                                        CUDA_R_32F, CUSPARSE_ORDER_COL) );
    // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, matA, matB, &beta, matC, CUDA_R_32F,
            CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) );
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );

    // execute SpMM
    CHECK_CUSPARSE( cusparseSpMM(handle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer) );

    clock_t tSPMM = clock();

    // device result check

    double hCsrValC[nnzc];
    int hCsrRowPtrC[m + 1];
    int hCsrColIndC[nnzc];

    CHECK_CUDA( cudaMemcpy(hCsrValC, dCsrValC, nnzc * sizeof(double), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrRowPtrC, dCsrRowPtrC, (m + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CHECK_CUDA( cudaMemcpy(hCsrColIndC, dCsrColIndC, nnzc * sizeof(int), cudaMemcpyDeviceToHost) );

    clock_t tCopied2Host = clock();

    // step 6: free resources

    // device memory deallocation
    // destroy matrix/vector descriptors
    CHECK_CUSPARSE( cusparseDestroySpMat(matA) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matB) );
    CHECK_CUSPARSE( cusparseDestroyDnMat(matC) );
    CHECK_CUSPARSE(cusparseDestroyMatDescr(descrC));
    CHECK_CUDA(cudaFree(dCsrValC) );
    CHECK_CUDA(cudaFree(dCsrRowPtrC) );
    CHECK_CUDA(cudaFree(dCsrColIndC) );
    CHECK_CUDA(cudaFree(dA) );
    CHECK_CUDA(cudaFree(dB) );
    CHECK_CUDA(cudaFree(dC) );

    // destroy
    CHECK_CUSPARSE(cusparseDestroy(handle));

    clock_t tEnd = clock();

    printf("\nTime to copy to device:\t\t\t\t%fms", (double)(tCopied2Device - tStart)/(CLOCKS_PER_SEC/1000));
    printf("\nTime to convert matrix from dense to sparse:\t%fms", (double)(tDense2Csr - tCopied2Device)/(CLOCKS_PER_SEC/1000));
    printf("\nTime to execute SPMM operation:\t\t\t%fms", (double)(tSPMM - tDense2Csr)/(CLOCKS_PER_SEC/1000));
    printf("\nTime to copy result to host:\t\t\t%fms", (double)(tCopied2Host - tSPMM)/(CLOCKS_PER_SEC/1000));
    printf("\nTime to free resources:\t\t\t\t%fms", (double)(tEnd - tCopied2Host)/(CLOCKS_PER_SEC/1000));
    printf("\nTotal time:\t\t\t\t\t%fms\n", (double)(tEnd - tStart)/(CLOCKS_PER_SEC/1000));

    return EXIT_SUCCESS;
}
