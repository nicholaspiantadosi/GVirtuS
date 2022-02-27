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
