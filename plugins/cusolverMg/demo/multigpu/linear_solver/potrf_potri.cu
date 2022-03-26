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

// #define SHOW_FORMAT

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

/* compute |x|_inf */
static double vec_nrm_inf(
        int n,
        const double *x)
{
    double max_nrm = 0;
    for(int row = 1; row <= n ; row++){
        double xi = x[ IDX1F(row) ];
        max_nrm = ( max_nrm > fabs(xi) )? max_nrm : fabs(xi);
    }
    return max_nrm;
}

/* A is 1D laplacian, return A(N:-1:1, :) */
static void gen_1d_laplacian(
        int N,
        double *A,
        int lda)
{
    memset(A, 0, sizeof(double)*lda*N);
    for(int J = 1 ; J <= N; J++ ){
        A[ IDX2F( J, J, lda ) ] = 2.0;
        if ( (J-1) >= 1 ){
            A[ IDX2F( J, J-1, lda ) ] = -1.0;
        }
        if ( (J+1) <= N ){
            A[ IDX2F( J, J+1, lda ) ] = -1.0;
        }
    }
}

/* Generate matrix B := A * X */
static void gen_ref_B(
        int N,
        int NRHS,
        double *A,
        int lda,
        double *X,
        int ldx,
        double *B,
        int ldb)
{
    memset(B, 0, sizeof(double)*lda*NRHS);

    for(int J = 1 ; J <= NRHS; J++ ){
        for(int I = 1 ; I <= N; I++ ){
            for(int K = 1 ; K <= N; K++ ){
                double Aik = A[ IDX2F( I, K, lda ) ];
                double  Xk = X[ IDX2F( K, J, ldx ) ];
                B[ IDX2F( I, J, ldb ) ] += (Aik * Xk);
            }
        }
    }
}

/* Apply inverse to RHS matrix */
static void solve_system_with_invA(
        int N,
        int NRHS,
        double *A,
        int lda,
        double *B,
        int ldb,
        double *X,
        int ldx)
{
    /* Extend lower triangular A to full matrix */
    for(int I = 1 ; I <= N; I++ ){
        for(int J = (I +1) ; J <= N; J++ ){
            A[ IDX2F( I, J, lda) ] = A[ IDX2F( J, I, lda) ];
        }
    }

#ifdef SHOW_FORMAT
    print_matrix( N, N, A, lda, "full inv(A)");
#endif

    /* Reset input matrix to 0 */
    memset(X, 0, sizeof(double)*lda*NRHS);

    /* Apply full inv(A) by matrix-matrix multiplication */

    for(int J = 1 ; J <= NRHS; J++ ){
        for(int I = 1 ; I <= N; I++ ){
            for(int K = 1 ; K <= N; K++ ){
                double Aik = A[ IDX2F( I, K, lda ) ];
                double  Bk = B[ IDX2F( K, J, ldx ) ];
                X[ IDX2F( I, J, ldx ) ] += (Aik * Bk);
            }
        }
    }
};


int main( int argc, char* argv[])
{
    cusolverMgHandle_t handle = NULL;
    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat = cudaSuccess;
/* maximum local number of GPUs, set by user */
    const int MAX_NUM_DEVICES = 2;

    int nbGpus = 0;
    int deviceList[MAX_NUM_DEVICES];
    const int NRHS = 2;
    const int N    = 8;

    const int IA  = 1;
    const int JA  = 1;
    const int T_A = 256; /* tile size of A */
    const int lda = N;

    const int IB  = 1;
    const int JB  = 1;
    const int T_B = 10; /* tile size of B */
    const int ldb = N;

    double *A = NULL;    /* A is N-by-N    */
    double *B = NULL;    /* B is N-by-NRHS */
    double *Xref = NULL; /* X is N-by-NRHS */
    double *Xans = NULL; /* X is N-by-NRHS */
    int  info = 0;

    cudaLibMgMatrixDesc_t descrA;
    cudaLibMgGrid_t gridA;
    cusolverMgGridMapping_t mapping = CUDALIBMG_GRID_MAPPING_COL_MAJOR;

    double **array_d_A = NULL;

    int64_t lwork_potrf = 0 ;
    int64_t lwork_potri = 0 ;
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
    B = (double *)malloc (sizeof(double)*ldb*NRHS);
    Xref = (double *)malloc (sizeof(double)*ldb*NRHS);
    Xans = (double *)malloc (sizeof(double)*ldb*NRHS);
    assert( NULL != A );
    assert( NULL != B );
    assert( NULL != Xref );
    assert( NULL != Xans );

/* permute 1D Laplacian to enable pivoting */
    printf("step 4: prepare 1D Laplacian for A and Xref = ones(N,NRHS) \n");
    gen_1d_laplacian(
            N,
            &A[ IDX2F( IA, JA, lda ) ],
            lda
    );

#ifdef SHOW_FORMAT
    print_matrix( N, N, A, lda, "A");
#endif

    /* X = ones(N,1) */
    for(int row = 1 ; row <= N ; row++){
        for(int col = 1 ; col <= NRHS ; col++){
            Xref[IDX2F(row, col, ldb)] = 1.0;
        }
    }
#ifdef SHOW_FORMAT
    print_matrix( N, NRHS, Xref, ldb, "Reference solution (X)");
#endif


/* Set B := A * X */
    printf("step 5: create rhs for reference solution on host B = A*Xref \n");
    gen_ref_B (
            N,
            NRHS,
            A,   /* input */
            lda,
            Xref,   /* input */
            ldb, /* same leading dimension as B */
            B,   /* output */
            ldb);

#ifdef SHOW_FORMAT
    print_matrix( N, NRHS, B, ldb, "Generated rhs (B)");
#endif
    printf("step 6: create matrix descriptors for A and B \n");
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

    printf("step 7: allocate distributed matrices A and B \n");
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

    printf("step 8: prepare data on devices \n");
/* distribute A to array_d_A */
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

    printf("step 9: allocate workspace space \n");
    status = cusolverMgPotrf_bufferSize(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            N,
            (void**)array_d_A,
            IA, /* base-1 */
            JA, /* base-1 */
            descrA,
            CUDA_R_64F,
            &lwork_potrf);
    assert(CUSOLVER_STATUS_SUCCESS == status);
    status = cusolverMgPotri_bufferSize(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            N,
            (void**)array_d_A,
            IA,
            JA,
            descrA,
            CUDA_R_64F,
            &lwork_potri);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    lwork = (lwork_potrf > lwork_potri)? lwork_potrf : lwork_potri;
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
    cudaStat = cudaDeviceSynchronize(); /* sync all devices */
    assert(cudaSuccess == cudaStat);

    printf("step 10: solve A*X = B by POTRF and POTRI \n");
    status = cusolverMgPotrf(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            N,
            (void**)array_d_A,
            IA,
            JA,
            descrA,
            CUDA_R_64F,
            (void**)array_d_work,
            lwork,
            &info  /* host */
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat = cudaDeviceSynchronize(); /* sync all devices */
    assert(cudaSuccess == cudaStat);
    assert(0 == info); /* check if A is singular  */
    status = cusolverMgPotri(
            handle,
            CUBLAS_FILL_MODE_LOWER,
            N,
            (void**)array_d_A,
            IA,
            JA,
            descrA,
            CUDA_R_64F,
            (void**)array_d_work,
            lwork,
            &info  /* host */
    );
    assert(CUSOLVER_STATUS_SUCCESS == status);
    cudaStat = cudaDeviceSynchronize(); /* sync all devices */
    assert(cudaSuccess == cudaStat);
    assert(0 == info); /* check if parameters are valid */

    printf("step 11: Gather inv(A) from devices to host \n");
    memcpyD2H<double>(
            nbGpus,
            deviceList,
            N,
            N,
/* input */
            N,   /* number of columns of global B */
            T_A, /* number of columns per column tile */
            ldb, /* leading dimension of local B */
            array_d_A,
            IA,
            JA,
/* output */
            A,   /* N-by-1 */
            ldb
    );

#ifdef SHOW_FORMAT
    /* A is N-by-N*/
    print_matrix(N, N, A, lda, "Computed solution inv(A)");
#endif

    printf("step 12: solve linear system B := inv(A) * B \n");
    solve_system_with_invA(
            N,
            NRHS,
            A,
            lda,
            B,
            ldb,
            Xans,
            ldb);
#ifdef SHOW_FORMAT
    /* A is N-by-N*/
    print_matrix(N, NRHS, Xans, ldb, "Computed solution Xans");
#endif


    printf("step 13: measure residual error |Xref - Xans| \n");
    double max_err = 0.0;
    for(int col = 1; col <= NRHS ; col++){
        printf("errors for X[:,%d] \n", col);
        double err = 0.0; /* absolute error per column */
        for(int row = 1; row <= N ; row++){
            double  Xref_ij = Xref[ IDX2F(row, col, ldb) ];
            double  Xans_ij = Xans[ IDX2F(row, col, ldb) ];
            double  err = fabs(Xref_ij - Xans_ij);
            max_err = ( err > max_err ) ? err : max_err;
        }
        double Xref_nrm_inf = vec_nrm_inf(N, &Xref[ IDX2F( 1, col, ldb)]);
        double Xans_nrm_inf = vec_nrm_inf(N, &Xans[ IDX2F( 1, col, ldb)]);
        double A_nrm_inf = 4.0;
        double rel_err = max_err/(A_nrm_inf * Xans_nrm_inf + Xref_nrm_inf);
        printf("\t|b - A*x|_inf = %E\n", max_err);
        printf("\t|Xref|_inf = %E\n", Xref_nrm_inf);
        printf("\t|Xans|_inf = %E\n", Xans_nrm_inf);
        printf("\t|A|_inf = %E\n", A_nrm_inf);
        /* relative error is around machine zero  */
        /* the user can use |b - A*x|/(N*|A|*|x|+|b|) as well */
        printf("\t|b - A*x|/(|A|*|x|+|b|) = %E\n\n", rel_err);
    }

    printf("step 14: free resources \n");
    destroyMat(
            nbGpus,
            deviceList,
            N,   /* number of columns of global A */
            T_A, /* number of columns per column tile */
            (void**)array_d_A );

    workspaceFree( nbGpus, deviceList, (void**)array_d_work );

    if (NULL != A) free(A);
    if (NULL != B) free(B);
    if (NULL != Xref) free(Xref);
    if (NULL != Xans) free(Xans);

    if (NULL != array_d_A   ) free(array_d_A);
    if (NULL != array_d_work) free(array_d_work);

    return 0;
}