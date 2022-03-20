#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverRf.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include "../cusolver_utils.h"

int main(void) {

    cusolverRfHandle_t handle = NULL;

    cusolverStatus_t cs = cusolverRfCreate(&handle);
    int correct = 1;
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int n = 3;
    int nnzA = 9;
    double hCsrValA[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA[] = {0, 3, 6, 9};
    const int hCsrColIndA[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzL = 9;
    double hCsrValL[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrL[] = {0, 3, 6, 9};
    const int hCsrColIndL[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int nnzU = 9;
    double hCsrValU[] = {10, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrU[] = {0, 3, 6, 9};
    const int hCsrColIndU[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int hP[] = {1, 1, 1};
    int hQ[] = {1, 1, 1};
    double *dCsrValA, *dCsrValL, *dCsrValU;
    int *dCsrRowPtrA, *dCsrColIndA, *dCsrRowPtrL, *dCsrColIndL, *dCsrRowPtrU, *dCsrColIndU, *dP, *dQ;
    CUDA_CHECK( cudaMalloc((void**) &dCsrValA, nnzA * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrA, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndA, nnzA * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrValL, nnzL * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrL, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndL, nnzL * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrValU, nnzU * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrU, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndU, nnzU * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dP, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dQ, n * sizeof(int)));
    CUDA_CHECK( cudaMemcpy(dCsrValA, hCsrValA, nnzA * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrA, hCsrRowPtrA, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndA, hCsrColIndA, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrValL, hCsrValL, nnzL * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrL, hCsrRowPtrL, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndL, hCsrColIndL, nnzL * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrValU, hCsrValU, nnzU * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrU, hCsrRowPtrU, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndU, hCsrColIndU, nnzU * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dP, hP, n * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dQ, hQ, n * sizeof(int), cudaMemcpyHostToDevice) );
    cs = cusolverRfSetupDevice(n, nnzA, dCsrRowPtrA, dCsrColIndA, dCsrValA, nnzL, dCsrRowPtrL, dCsrColIndL, dCsrValL, nnzU, dCsrRowPtrU, dCsrColIndU, dCsrValU, dP, dQ, handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfAnalyze(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfRefactor(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int nnzM;
    int *dMp, *dMi;
    double *dMx;
    CUDA_CHECK( cudaMalloc((void**) &dMp, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dMi, nnzM * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dMx, nnzM * sizeof(double)));
    cs = cusolverRfAccessBundledFactorsDevice(handle, &nnzM, &dMp, &dMi, &dMx);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("nnzM: %d\n", nnzM);
    correct = nnzM == 15;
    int hMp[n + 1];
    int hMi[nnzM];
    double hMx[nnzM];
    int hMp_result[] = {0, 5, 10, 15};
    int hMi_result[] = {0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2};
    double hMx_result[] = {0.000000, 0.000000, 10.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000};
    CUDA_CHECK( cudaMemcpy(hMp, dMp, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(hMi, dMi, nnzM * sizeof(int), cudaMemcpyDeviceToHost) );
    CUDA_CHECK( cudaMemcpy(hMx, dMx, nnzM * sizeof(double), cudaMemcpyDeviceToHost) );
    //printArray(hMp, (n+1), "Mp");
    //printArray(hMi, nnzM, "Mi");
    //printArrayD(hMx, nnzM, "Mx");
    for (int i = 0; i < (n + 1); i++) {
        if (fabsf(hMp[i] - hMp_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }
    for (int i = 0; i < nnzM; i++) {
        if (fabsf(hMi[i] - hMi_result[i]) > 0.001 || fabsf(hMx[i] - hMx_result[i]) > 0.001) {
            correct = 0;
            break;
        }
    }

    cusolverRfMatrixFormat_t format;
    cusolverRfUnitDiagonal_t diag;
    cs = cusolverRfGetMatrixFormat(handle, &format, &diag);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("format: %d\n", format);
    correct = format == CUSOLVERRF_MATRIX_FORMAT_CSR;
    //printf("diag: %d\n", diag);
    correct = diag == CUSOLVERRF_UNIT_DIAGONAL_STORED_L;

    cs = cusolverRfSetMatrixFormat(handle, CUSOLVERRF_MATRIX_FORMAT_CSC, CUSOLVERRF_UNIT_DIAGONAL_STORED_U);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfGetMatrixFormat(handle, &format, &diag);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    correct = format == CUSOLVERRF_MATRIX_FORMAT_CSC;
    correct = diag == CUSOLVERRF_UNIT_DIAGONAL_STORED_U;

    cs = cusolverRfSetMatrixFormat(handle, CUSOLVERRF_MATRIX_FORMAT_CSR, CUSOLVERRF_UNIT_DIAGONAL_STORED_L);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double zero;
    double boost;
    cs = cusolverRfGetNumericProperties(handle, &zero, &boost);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("zero: %f\n", zero);
    correct = zero == 0;
    //printf("boost: %f\n", boost);
    correct = boost == 0;

    cs = cusolverRfSetNumericProperties(handle, 1, 1);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfGetNumericProperties(handle, &zero, &boost);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    correct = zero == 1;
    correct = boost == 1;

    cs = cusolverRfSetNumericProperties(handle, 0, 0);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolverRfNumericBoostReport_t report;
    cs = cusolverRfGetNumericBoostReport(handle, &report);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("report: %d\n", report);
    correct = report == CUSOLVERRF_NUMERIC_BOOST_NOT_USED;

    cusolverRfResetValuesFastMode_t fastMode;
    cs = cusolverRfGetResetValuesFastMode(handle, &fastMode);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("fastMode: %d\n", fastMode);
    correct = fastMode == CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF;

    cs = cusolverRfSetResetValuesFastMode(handle, CUSOLVERRF_RESET_VALUES_FAST_MODE_ON);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfGetResetValuesFastMode(handle, &fastMode);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    correct = fastMode == CUSOLVERRF_RESET_VALUES_FAST_MODE_ON;

    cs = cusolverRfSetResetValuesFastMode(handle, CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolverRfFactorization_t fact_alg;
    cusolverRfTriangularSolve_t solve_alg;
    cs = cusolverRfGetAlgs(handle, &fact_alg, &solve_alg);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    //printf("fact_alg: %d\n", fact_alg);
    correct = fact_alg == CUSOLVERRF_FACTORIZATION_ALG0;
    //printf("solve_alg: %d\n", solve_alg);
    correct = solve_alg == CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;

    cs = cusolverRfSetAlgs(handle, CUSOLVERRF_FACTORIZATION_ALG1, CUSOLVERRF_TRIANGULAR_SOLVE_ALG2);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfGetAlgs(handle, &fact_alg, &solve_alg);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }
    correct = fact_alg == CUSOLVERRF_FACTORIZATION_ALG1;
    correct = solve_alg == CUSOLVERRF_TRIANGULAR_SOLVE_ALG2;

    cs = cusolverRfSetAlgs(handle, CUSOLVERRF_FACTORIZATION_ALG0, CUSOLVERRF_TRIANGULAR_SOLVE_ALG1);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int nrhs = 3;
    int ldt = 3;
    int ldxf = 3;
    double *Temp, *XF;
    CUDA_CHECK( cudaMalloc((void**) &Temp, ldt * nrhs * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &XF, ldxf * nrhs * sizeof(double)));
    cs = cusolverRfSolve(handle, dP, dQ, 1, Temp, ldt, XF, ldxf);
    double hXF[ldxf * nrhs];
    CUDA_CHECK( cudaMemcpy(hXF, XF, ldxf * nrhs * sizeof(double), cudaMemcpyDeviceToHost) );

    double hCsrValA2[] = {20, 1, 9, 3, 4, -6, 1, 6, 2};
    const int hCsrRowPtrA2[] = {0, 3, 6, 9};
    const int hCsrColIndA2[] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
    int hP2[] = {2, 2, 2};
    int hQ2[] = {2, 2, 2};
    double *dCsrValA2;
    int *dCsrRowPtrA2, *dCsrColIndA2, *dP2, *dQ2;
    CUDA_CHECK( cudaMalloc((void**) &dCsrValA2, nnzA * sizeof(double)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrRowPtrA2, (n + 1) * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dCsrColIndA2, nnzA * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dP2, n * sizeof(int)));
    CUDA_CHECK( cudaMalloc((void**) &dQ2, n * sizeof(int)));
    CUDA_CHECK( cudaMemcpy(dCsrValA2, hCsrValA2, nnzA * sizeof(double), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrRowPtrA2, hCsrRowPtrA2, (n + 1) * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dCsrColIndA2, hCsrColIndA2, nnzA * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dP2, hP2, n * sizeof(int), cudaMemcpyHostToDevice) );
    CUDA_CHECK( cudaMemcpy(dQ2, hQ2, n * sizeof(int), cudaMemcpyHostToDevice) );
    cs = cusolverRfResetValues(n, nnzA, dCsrRowPtrA2, dCsrColIndA2, dCsrValA2, dP2, dQ2, handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverRfDestroy(handle);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    CUDA_CHECK(cudaFree(dCsrValA));
    CUDA_CHECK(cudaFree(dCsrRowPtrA));
    CUDA_CHECK(cudaFree(dCsrColIndA));
    CUDA_CHECK(cudaFree(dCsrValL));
    CUDA_CHECK(cudaFree(dCsrRowPtrL));
    CUDA_CHECK(cudaFree(dCsrColIndL));
    CUDA_CHECK(cudaFree(dCsrValU));
    CUDA_CHECK(cudaFree(dCsrRowPtrU));
    CUDA_CHECK(cudaFree(dCsrColIndU));
    CUDA_CHECK(cudaFree(dP));
    CUDA_CHECK(cudaFree(dQ));
    CUDA_CHECK(cudaFree(dCsrValA2));
    CUDA_CHECK(cudaFree(dCsrRowPtrA2));
    CUDA_CHECK(cudaFree(dCsrColIndA2));
    CUDA_CHECK(cudaFree(dP2));
    CUDA_CHECK(cudaFree(dQ2));

    if (correct == 1) {
        printf("rf_create_destroy test PASSED\n");
    } else {
        printf("rf_create_destroy test FAILED\n");
    }

    return EXIT_SUCCESS;
}