/*
 * gVirtuS -- A GPGPU transparent virtualization component.
 *
 * Copyright (C) 2009-2010  The University of Napoli Parthenope at Naples.
 *
 * This file is part of gVirtuS.
 *
 * gVirtuS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * gVirtuS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with gVirtuS; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 *
 * Written by: Antonio Pilato <antonio.pilato001@studenti.uniparthenope.it>,
 *             Nicholas Piantadosi <nicholas.piantadosi@studenti.uniparthenope.it>,
 *              Department of Science andTechnology
 */

#include <iostream>
#include <cstdio>
#include <string>

#include "CusolverFrontend.h"

using namespace std;

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreate(cusolverDnHandle_t *handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverDnHandle_t>(handle);
    CusolverFrontend::Execute("cusolverDnCreate");
    if(CusolverFrontend::Success())
    	*handle = CusolverFrontend::GetOutputVariable<cusolverDnHandle_t>();
    return CusolverFrontend::GetExitCode();
}
    
extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDestroy(cusolverDnHandle_t handle) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverDnDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSetStream(cusolverDnHandle_t handle, cudaStream_t streamId) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) streamId);
    CusolverFrontend::Execute("cusolverDnSetStream");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnGetStream(cusolverDnHandle_t handle, cudaStream_t *streamId) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::Execute("cusolverDnGetStream");
    if(CusolverFrontend::Success())
        *streamId = (cudaStream_t) CusolverFrontend::GetOutputVariable<size_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreateSyevjInfo(syevjInfo_t *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<syevjInfo_t>(info);
    CusolverFrontend::Execute("cusolverDnCreateSyevjInfo");
    if(CusolverFrontend::Success())
        *info = CusolverFrontend::GetOutputVariable<syevjInfo_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDestroySyevjInfo(syevjInfo_t info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnDestroySyevjInfo");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetTolerance(syevjInfo_t info, double tolerance) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<double>(tolerance);
    CusolverFrontend::Execute("cusolverDnXsyevjSetTolerance");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetMaxSweeps(syevjInfo_t info, int max_sweeps) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<int>(max_sweeps);
    CusolverFrontend::Execute("cusolverDnXsyevjSetMaxSweeps");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjSetSortEig(syevjInfo_t info, int sort_eig) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<int>(sort_eig);
    CusolverFrontend::Execute("cusolverDnXsyevjSetSortEig");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetResidual(cusolverDnHandle_t handle, syevjInfo_t info, double *residual) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnXsyevjGetResidual");
    if(CusolverFrontend::Success())
        *residual = CusolverFrontend::GetOutputVariable<double>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXsyevjGetSweeps(cusolverDnHandle_t handle, syevjInfo_t info, int *executed_sweeps) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnXsyevjGetSweeps");
    if(CusolverFrontend::Success())
        *executed_sweeps = CusolverFrontend::GetOutputVariable<int>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreateGesvdjInfo(gesvdjInfo_t *info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<gesvdjInfo_t>(info);
    CusolverFrontend::Execute("cusolverDnCreateGesvdjInfo");
    if(CusolverFrontend::Success())
        *info = CusolverFrontend::GetOutputVariable<gesvdjInfo_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDestroyGesvdjInfo(gesvdjInfo_t info) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnDestroyGesvdjInfo");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetTolerance(gesvdjInfo_t info, double tolerance) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<double>(tolerance);
    CusolverFrontend::Execute("cusolverDnXgesvdjSetTolerance");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetMaxSweeps(gesvdjInfo_t info, int max_sweeps) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<int>(max_sweeps);
    CusolverFrontend::Execute("cusolverDnXgesvdjSetMaxSweeps");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjSetSortEig(gesvdjInfo_t info, int sort_eig) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::AddVariableForArguments<int>(sort_eig);
    CusolverFrontend::Execute("cusolverDnXgesvdjSetSortEig");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetResidual(cusolverDnHandle_t handle, gesvdjInfo_t info, double *residual) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnXgesvdjGetResidual");
    if(CusolverFrontend::Success())
        *residual = CusolverFrontend::GetOutputVariable<double>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnXgesvdjGetSweeps(cusolverDnHandle_t handle, gesvdjInfo_t info, int *executed_sweeps) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) handle);
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) info);
    CusolverFrontend::Execute("cusolverDnXgesvdjGetSweeps");
    if(CusolverFrontend::Success())
        *executed_sweeps = CusolverFrontend::GetOutputVariable<int>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsCreate(cusolverDnIRSParams_t *params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverDnIRSParams_t>(params);
    CusolverFrontend::Execute("cusolverDnIRSParamsCreate");
    if(CusolverFrontend::Success())
        *params = CusolverFrontend::GetOutputVariable<cusolverDnIRSParams_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsDestroy(cusolverDnIRSParams_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnIRSParamsDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverPrecisions(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision, cusolverPrecType_t solver_lowest_precision) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverPrecType_t>(solver_main_precision);
    CusolverFrontend::AddVariableForArguments<cusolverPrecType_t>(solver_lowest_precision);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetSolverPrecisions");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverMainPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t solver_main_precision) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverPrecType_t>(solver_main_precision);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetSolverMainPrecision");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetSolverLowestPrecision(cusolverDnIRSParams_t params, cusolverPrecType_t lowest_precision_type) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverPrecType_t>(lowest_precision_type);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetSolverLowestPrecision");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetRefinementSolver(cusolverDnIRSParams_t params, cusolverIRSRefinement_t solver) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverIRSRefinement_t>(solver);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetRefinementSolver");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetTol(cusolverDnIRSParams_t params, double val) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<double>(val);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetTol");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetTolInner(cusolverDnIRSParams_t params, double val) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<double>(val);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetTolInner");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetMaxIters(cusolverDnIRSParams_t params, int max_iters) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<int>(max_iters);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetMaxIters");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsSetMaxItersInner(cusolverDnIRSParams_t params, cusolver_int_t maxiters_inner) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolver_int_t>(maxiters_inner);
    CusolverFrontend::Execute("cusolverDnIRSParamsSetMaxItersInner");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsEnableFallback(cusolverDnIRSParams_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnIRSParamsEnableFallback");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsDisableFallback(cusolverDnIRSParams_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnIRSParamsDisableFallback");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSParamsGetMaxIters(cusolverDnIRSParams_t params, cusolver_int_t *maxiters) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnIRSParamsGetMaxIters");
    if(CusolverFrontend::Success())
        *maxiters = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosCreate(cusolverDnIRSInfos_t *infos) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverDnIRSInfos_t>(infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosCreate");
    if(CusolverFrontend::Success())
        *infos = CusolverFrontend::GetOutputVariable<cusolverDnIRSInfos_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosDestroy(cusolverDnIRSInfos_t infos) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosDestroy");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetMaxIters(cusolverDnIRSInfos_t infos, cusolver_int_t *maxiters) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosGetMaxIters");
    if(CusolverFrontend::Success())
        *maxiters = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetNiters(cusolverDnIRSInfos_t infos, cusolver_int_t *niters) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosGetNiters");
    if(CusolverFrontend::Success())
        *niters = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetOuterNiters(cusolverDnIRSInfos_t infos, cusolver_int_t *outer_niters) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosGetOuterNiters");
    if(CusolverFrontend::Success())
        *outer_niters = CusolverFrontend::GetOutputVariable<cusolver_int_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosRequestResidual(cusolverDnIRSInfos_t infos) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosRequestResidual");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnIRSInfosGetResidualHistory(cusolverDnIRSInfos_t infos, void **residual_history) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) infos);
    CusolverFrontend::Execute("cusolverDnIRSInfosGetResidualHistory");
    if(CusolverFrontend::Success())
        *residual_history = (void *) CusolverFrontend::GetOutputDevicePointer();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnCreateParams(cusolverDnParams_t *params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddHostPointerForArguments<cusolverDnParams_t>(params);
    CusolverFrontend::Execute("cusolverDnCreateParams");
    if(CusolverFrontend::Success())
        *params = CusolverFrontend::GetOutputVariable<cusolverDnParams_t>();
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnDestroyParams(cusolverDnParams_t params) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::Execute("cusolverDnDestroyParams");
    return CusolverFrontend::GetExitCode();
}

extern "C" cusolverStatus_t CUSOLVERAPI cusolverDnSetAdvOptions(cusolverDnParams_t params, cusolverDnFunction_t function, cusolverAlgMode_t algo) {
    CusolverFrontend::Prepare();
    CusolverFrontend::AddVariableForArguments<size_t>((size_t) params);
    CusolverFrontend::AddVariableForArguments<cusolverDnFunction_t>(function);
    CusolverFrontend::AddVariableForArguments<cusolverAlgMode_t>(algo);
    CusolverFrontend::Execute("cusolverDnSetAdvOptions");
    return CusolverFrontend::GetExitCode();
}
