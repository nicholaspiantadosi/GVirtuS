/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include batchsyevj_example.cpp
 *   g++ -o batchsyevj_example batchsyevj_example.o -L/usr/local/cuda/lib64 -lcusolver -lcudart
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

int main(int argc, char*argv[])
{
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    syevjInfo_t syevj_params = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    const int m = 3;
    const int lda = m;
    const int batchSize = 2;
/*
 *        |  1  -1   0 |
 *   A0 = | -1   2   0 |
 *        |  0   0   0 |
 *
 *   A0 = V0 * W0 * V0**T
 *
 *   W0 = diag(0, 0.3820, 2.6180)
 *
 *        |  3   4  0 |
 *   A1 = |  4   7  0 |
 *        |  0   0  0 |
 *
 *   A1 = V1 * W1 * V1**T
 *
 *   W1 = diag(0, 0.5279, 9.4721)
 *
 */
    double A[lda*m*batchSize]; /* A = [A0 ; A1] */
    double V[lda*m*batchSize]; /* V = [V0 ; V1] */
    double W[m*batchSize];     /* W = [W0 ; W1] */
    int info[batchSize];       /* info = [info0 ; info1] */

    double *d_A  = NULL; /* lda-by-m-by-batchSize */
    double *d_W  = NULL; /* m-by-batchSizee */
    int* d_info  = NULL; /* batchSize */
    int lwork = 0;  /* size of workspace */
    double *d_work = NULL; /* device workspace for syevjBatched */

    const double tol = 1.e-7;
    const int max_sweeps = 15;
    const int sort_eig  = 0;   /* don't sort eigenvalues */
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; /* compute eigenvectors */
    const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;

/* residual and executed_sweeps are not supported on syevjBatched */
    double residual = 0;
    int executed_sweeps = 0;

    double *A0 = A;
    double *A1 = A + lda*m;
/*
 *        |  1  -1   0 |
 *   A0 = | -1   2   0 |
 *        |  0   0   0 |
 *   A0 is column-major
 */
    A0[0 + 0*lda] =  1.0;
    A0[1 + 0*lda] = -1.0;
    A0[2 + 0*lda] =  0.0;

    A0[0 + 1*lda] = -1.0;
    A0[1 + 1*lda] =  2.0;
    A0[2 + 1*lda] =  0.0;

    A0[0 + 2*lda] =  0.0;
    A0[1 + 2*lda] =  0.0;
    A0[2 + 2*lda] =  0.0;
/*
 *        |  3   4  0 |
 *   A1 = |  4   7  0 |
 *        |  0   0  0 |
 *   A1 is column-major
 */
    A1[0 + 0*lda] = 3.0;
    A1[1 + 0*lda] = 4.0;
    A1[2 + 0*lda] = 0.0;

    A1[0 + 1*lda] = 4.0;
    A1[1 + 1*lda] = 7.0;
    A1[2 + 1*lda] = 0.0;

    A1[0 + 2*lda] = 0.0;
    A1[1 + 2*lda] = 0.0;
    A1[2 + 2*lda] = 0.0;
    /* step 1: create cusolver handle, bind a stream  */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: configuration of syevj */
    status = cusolverDnCreateSyevjInfo(&syevj_params);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of tolerance is machine zero */
    status = cusolverDnXsyevjSetTolerance(
            syevj_params,
            tol);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* default value of max. sweeps is 100 */
    status = cusolverDnXsyevjSetMaxSweeps(
            syevj_params,
            max_sweeps);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* disable sorting */
    status = cusolverDnXsyevjSetSortEig(
            syevj_params,
            sort_eig);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 3: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A   , sizeof(double) * lda * m * batchSize);
    cudaStat2 = cudaMalloc ((void**)&d_W   , sizeof(double) * m * batchSize);
    cudaStat3 = cudaMalloc ((void**)&d_info, sizeof(int   ) * batchSize);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m * batchSize, cudaMemcpyHostToDevice);
    cudaStat2 = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

/* step 4: query working space of syevjBatched */
    status = cusolverDnDsyevjBatched_bufferSize(
            cusolverH,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_W,
            &lwork,
            syevj_params,
            batchSize
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);
    /* step 5: compute spectrum of A0 and A1 */
    status = cusolverDnDsyevjBatched(
            cusolverH,
            jobz,
            uplo,
            m,
            d_A,
            lda,
            d_W,
            d_work,
            lwork,
            d_info,
            syevj_params,
            batchSize
    );
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(V    , d_A   , sizeof(double) * lda * m * batchSize, cudaMemcpyDeviceToHost);
    cudaStat2 = cudaMemcpy(W    , d_W   , sizeof(double) * m * batchSize      , cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int) * batchSize             , cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    for(int i = 0 ; i < batchSize ; i++){
        if ( 0 == info[i] ){
            printf("matrix %d: syevj converges \n", i);
        }else if ( 0 > info[i] ){
/* only info[0] shows if some input parameter is wrong.
 * If so, the error is CUSOLVER_STATUS_INVALID_VALUE.
 */
            printf("Error: %d-th parameter is wrong \n", -info[i] );
            exit(1);
        }else { /* info = m+1 */
/* if info[i] is not zero, Jacobi method does not converge at i-th matrix. */
            printf("WARNING: matrix %d, info = %d : sygvj does not converge \n", i, info[i] );
        }
    }

/* Step 6: show eigenvalues and eigenvectors */
    double *W0 = W;
    double *W1 = W + m;
    printf("==== \n");
    for(int i = 0 ; i < m ; i++){
        printf("W0[%d] = %f\n", i, W0[i]);
    }
    printf("==== \n");
    for(int i = 0 ; i < m ; i++){
        printf("W1[%d] = %f\n", i, W1[i]);
    }
    printf("==== \n");

    double *V0 = V;
    double *V1 = V + lda*m;
    printf("V0 = (matlab base-1)\n");
    printMatrix(m, m, V0, lda, "V0");
    printf("V1 = (matlab base-1)\n");
    printMatrix(m, m, V1, lda, "V1");
    /*
 * The folowing two functions do not support batched version.
 * The error CUSOLVER_STATUS_NOT_SUPPORTED is returned.
 */
    status = cusolverDnXsyevjGetSweeps(
            cusolverH,
            syevj_params,
            &executed_sweeps);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

    status = cusolverDnXsyevjGetResidual(
            cusolverH,
            syevj_params,
            &residual);
    assert(CUSOLVER_STATUS_NOT_SUPPORTED == status);

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_W    ) cudaFree(d_W);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);
    if (syevj_params) cusolverDnDestroySyevjInfo(syevj_params);

    cudaDeviceReset();

    return 0;
}