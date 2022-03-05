#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusolverDn.h>         // cusolverDn
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE

int main(void) {

    cusolverDnHandle_t handle = NULL;
    cusolverDnIRSParams_t params = NULL;

    cusolverStatus_t cs = cusolverDnCreate(&handle);

    int correct = 1;

    cs = cusolverDnIRSParamsCreate(&params);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsSetSolverPrecisions(params, CUSOLVER_R_64F, CUSOLVER_R_16F);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsSetSolverMainPrecision(params, CUSOLVER_R_64F);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsSetSolverLowestPrecision(params, CUSOLVER_R_16F);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsSetRefinementSolver(params, CUSOLVER_IRS_REFINE_CLASSICAL);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double tolerance = 0.1;
    cs = cusolverDnIRSParamsSetTol(params, tolerance);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    double toleranceInner = 0.1;
    cs = cusolverDnIRSParamsSetTolInner(params, toleranceInner);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    int max_iters = 40;
    cs = cusolverDnIRSParamsSetMaxIters(params, max_iters);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolver_int_t maxiters_inner = 30;
    cs = cusolverDnIRSParamsSetMaxItersInner(params, maxiters_inner);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsEnableFallback(params);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsDisableFallback(params);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolver_int_t max_iters_get = 0;
    cs = cusolverDnIRSParamsGetMaxIters(params, &max_iters_get);
    if (cs != CUSOLVER_STATUS_SUCCESS || max_iters_get != max_iters) {
        correct = 0;
    }

    cusolverDnIRSInfos_t infos = NULL;
    cs = cusolverDnIRSInfosCreate(&infos);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolver_int_t maxiters_get_2 = 0;
    cs = cusolverDnIRSInfosGetMaxIters(infos, &maxiters_get_2);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolver_int_t niters = 0;
    cs = cusolverDnIRSInfosGetMaxIters(infos, &niters);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cusolver_int_t outer_niters = 0;
    cs = cusolverDnIRSInfosGetMaxIters(infos, &outer_niters);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSInfosRequestResidual(infos);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    /*
    void* residual_history;
    cs = cusolverDnIRSInfosGetResidualHistory(infos, &residual_history);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
        printf("%d\n", cs);
    }
    */

    cusolverDnParams_t dnParams = NULL;
    cs = cusolverDnCreateParams(&dnParams);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnSetAdvOptions(dnParams, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnDestroyParams(dnParams);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSInfosDestroy(infos);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    cs = cusolverDnIRSParamsDestroy(params);
    if (cs != CUSOLVER_STATUS_SUCCESS) {
        correct = 0;
    }

    if (correct == 1) {
        printf("cusolver_dnirsparams test PASSED\n");
    } else {
        printf("cusolver_dnirsparams test FAILED\n");
    }

    cusolverDnDestroy(handle);

    return EXIT_SUCCESS;

}