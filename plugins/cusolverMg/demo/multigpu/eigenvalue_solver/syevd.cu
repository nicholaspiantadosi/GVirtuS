/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *       nvcc -ccbin gcc -I/usr/local/cuda/include  -c main.cpp -o main.o
 *       nvcc -cudart static main.o -lcusolverMg
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>
#include "cusolverMg.h"
#include "util.hxx"

//#define SHOW_FORMAT

#ifndef IDX2F
#define IDX2F(i,j,lda) ((((j)-1)*((size_t)lda))+((i)-1))
#endif /* IDX2F */

#ifndef IDX1F
#define IDX1F(i) ((i)-1)
#endif /* IDX1F */

static void print_matrix(
        int m,
        int n,
        const double *A,
        int lda,
        const char* name)
{
    printf("%s = matlab base-1, %d-by-%d matrix\n", name, m, n);
    for(int row = 1 ; row <= m ; row++){
        for(int col = 1 ; col <= n ; col++){
            double Aij = A[IDX2F(row, col, lda)];
            printf("%s(%d,%d) = %20.16E\n", name, row, col, Aij );
        }
    }
}

static void gen_1d_laplacian(
        int N,
        double *A,
        int lda)
{
    memset(A, 0, sizeof(double)*lda*N);
    for(int J = 1 ; J <= N; J++ ){
        /* A(J,J) = 2 */
        A[ IDX2F( J, J, lda ) ] = 2.0;
        if ( (J-1) >= 1 ){
            /* A(J, J-1) = -1*/
            A[ IDX2F( J, J-1, lda ) ] = -1.0;
        }
        if ( (J+1) <= N ){
            /* A(J, J+1) = -1*/
            A[ IDX2F( J, J+1, lda ) ] = -1.0;
        }
    }
}

int main( int argc, char* argv[])
{
    cusolverMgHandle_t handle = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;

/* maximum number of GPUs */
    const int MAX_NUM_DEVICES = 16;

    int nbGpus = 0;
    int deviceList[MAX_NUM_DEVICES];

    const int N   = 2111;
    const int IA  = 1;
    const int JA  = 1;
    const int T_A = 256; /* tile size */
    const int lda = N;

    double *A = NULL; /* A is N-by-N */
    double *D = NULL; /* D is 1-by-N */
    int  info = 0;

    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgGrid_t gridA;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    double **array_d_A = NULL;

    int64_t lwork = 0 ; /* workspace: number of elements per device */
    double **array_d_work = NULL;

    printf("test 1D Laplacian of order %d\n", N);

    printf("step 1: create Mg handle and select devices \n");
    status = cusolverMgCreate(&handle);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat = cudaGetDeviceCount( &nbGpus );
    assert( cudaSuccess == cudaStat );

    nbGpus = (nbGpus < MAX_NUM_DEVICES)? nbGpus : MAX_NUM_DEVICES;
    printf("\tthere are %d GPUs \n", nbGpus);
    for(int j = 0 ; j < nbGpus ; j++){
        deviceList[j] = j;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, j);
        printf("\tdevice %d, %s, cc %d.%d \n",j, prop.name, prop.major, prop.minor);
    }

    status = cusolverMgDeviceSelect(
            handle,
            nbGpus,
            deviceList);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("step 2: Enable peer access.\n");
    assert( 0 == enablePeerAccess( nbGpus, deviceList ) );
    printf("step 3: allocate host memory A \n");
    A = (double *)malloc (sizeof(double)*lda*N);
    D = (double *)malloc (sizeof(double)*N);
    assert( NULL != A );
    assert( NULL != D );

    printf("step 4: prepare 1D Laplacian \n");
    gen_1d_laplacian(
            N,
            &A[ IDX2F( IA, JA, lda ) ],
            lda
    );

#ifdef SHOW_FORMAT
    print_matrix( N, N, A, lda, "A");
#endif

    printf("step 5: create matrix descriptors for A and D \n");

    status = cusolverMgCreateDeviceGrid(&gridA, 1, nbGpus, deviceList, mapping );
    assert(CUSOLVER_STATUS_SUCCESS == status);
/* (global) A is N-by-N */
    status = cusolverMgCreateMatrixDesc(
            &descrA,
            N,   /* nubmer of rows of (global) A */
            N,   /* number of columns of (global) A */
            N,   /* number or rows in a tile */
            T_A, /* number of columns in a tile */
            CUDA_R_64F,
            gridA );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("step 6: allocate distributed matrices A and D \n");

    array_d_A = (double**)malloc(sizeof(double*)*nbGpus);
    assert(NULL != array_d_A);
/* A := 0 */
    createMat<double>(
            nbGpus,
            deviceList,
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            lda, /* leading dimension of local A */
            array_d_A
    );

    printf("step 7: prepare data on devices \n");
    memcpyH2D<double>(
            nbGpus,
            deviceList,
            N,
            N,
/* input */
            A,
            lda,
/* output */
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            lda, /* leading dimension of local A */
            array_d_A,   /* host pointer array of dimension nbGpus */
            IA,
            JA
    );
    printf("step 8: allocate workspace space \n");
    status = cusolverMgSyevd_bufferSize(
            handle,
            (cusolverEigMode_t)jobz,
            CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
            N,
            (void**)array_d_A,
            IA, /* base-1 */
            JA, /* base-1 */
            descrA,
            (void*)D,
            CUDA_R_64F,
            CUDA_R_64F,
            &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    printf("\tallocate device workspace, lwork = %lld \n", (long long)lwork);
    array_d_work = (double**)malloc(sizeof(double*)*nbGpus);
    assert( NULL != array_d_work);
/* array_d_work[j] points to device workspace of device j */
    workspaceAlloc(
            nbGpus,
            deviceList,
            sizeof(double)*lwork, /* number of bytes per device */
            (void**)array_d_work
    );

/* sync all devices */
    cudaStat = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat);

    printf("step 9: compute eigenvalues and eigenvectors \n");
    status = cusolverMgSyevd(
            handle,
            (cusolverEigMode_t)jobz,
            CUBLAS_FILL_MODE_LOWER, /* only support lower mode */
            N,
            (void**)array_d_A,  /* exit: eigenvectors */
            IA,
            JA,
            descrA,
            (void**)D,  /* exit: eigenvalues */
            CUDA_R_64F,
            CUDA_R_64F,
            (void**)array_d_work,
            lwork,
            &info  /* host */
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);

    /* sync all devices */
    cudaStat = cudaDeviceSynchronize();
    assert(cudaSuccess == cudaStat);

/* check if SYEVD converges */
    assert(0 == info);
    printf("step 10: copy eigenvectors to A and eigenvalues to D\n");

    memcpyD2H<double>(
            nbGpus,
            deviceList,
            N,
            N,
/* input */
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            lda, /* leading dimension of local A */
            array_d_A,
            IA,
            JA,
/* output */
            A,   /* N-y-N eigenvectors */
            lda
    );

#ifdef SHOW_FORMAT
    printf("eigenvalue D = \n");
    /* D is 1-by-N */
    print_matrix(1, N, D, 1, "D");
#endif

    printf("step 11: verify eigenvales \n");
    printf("     lambda(k) = 4 * sin(pi/2 *k/(N+1))^2 for k = 1:N \n");
    double max_err_D = 0;
    for(int k = 1; k <= N ; k++){
        const double pi = 4*atan(1.0);
        const double h  = 1.0/((double)N+1);
        const double factor = sin(pi/2.0 * ((double)k)*h);
        const double lambda = 4.0*factor*factor;
        const double err = fabs(D[IDX1F(k)] - lambda);
        max_err_D = (max_err_D > err)? max_err_D : err;
//        printf("k = %d, D = %E, lambda = %E, err = %E\n", k, D[IDX1F(k)], lambda, err);
    }
    printf("\n|D - lambda|_inf = %E\n\n", max_err_D);


    printf("step 12: free resources \n");
    destroyMat(
            nbGpus,
            deviceList,
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            (void**)array_d_A );

    workspaceFree( nbGpus, deviceList, (void**)array_d_work );

    if (NULL != A) free(A);
    if (NULL != D) free(D);

    if (NULL != array_d_A   ) free(array_d_A);
    if (NULL != array_d_work) free(array_d_work);

    return 0;
}